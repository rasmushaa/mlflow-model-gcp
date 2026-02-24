import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, partial, tpe
from hyperopt.pyll import scope
from metrics_toolbox import EvaluatorBuilder

from src.training import kfold_iterator

logger = logging.getLogger(__name__)


class TuneableModel(Protocol):
    def fit(self, X, y): ...
    def predict(self, X): ...
    def predict_proba(self, X): ...
    @classmethod
    def from_config(cls, cfg: dict) -> "TuneableModel": ...
    @property
    def classes(self): ...


@dataclass
class OptimizationResults:
    model: TuneableModel
    best_params: dict
    loss_history: list[float]
    metrics: dict


def is_spec(x):
    """Check if Pipeline hyperparameter spec is tunable.

    The pipeline config can contain hardcoded values for hyperparameters, e.g. "max_depth": 5,
    or tunable specs, e.g. "max_depth": {"type": "int", "min": 1, "max": 10}.

    Returns
    -------
    bool
        True if x is a tunable hyperparameter spec, False otherwise.
    """
    return (
        isinstance(x, dict)
        and "type" in x
        and any(k in x for k in ("min", "max", "choices"))
    )


def build_search_space(base_cfg: dict) -> dict:
    """Build hyperopt search space from pipeline config.

    Scans through the pipeline config for any hyperparameter specs
    and constructs a corresponding hyperopt search space.
    Only parameters with a valid spec (type + range/choices) are included in the search space,
    while others are ignored (treated as fixed values).
    Config "path" is used as key to prevent name clashes
    and to allow setting values back into the config after optimization.

    The hyperopt config must contain:
    - "type": one of "int", "float", "choice"
    - For "int" and "float": "min" and "max" (and optional "step")
    - For "choice": "choices" list

    If no valid tunable parameters are found,
    the function will return an empty search space.

    Parameters
    ----------
    base_cfg : dict
        The base pipeline config: {<component>: {"hyperparams": {<param>: spec_or_value}}, ...}
    """
    space = {}

    for step_name, step_cfg in base_cfg.items():
        hp_cfg = (step_cfg or {}).get("hyperparams", {}) or {}
        for param_name, spec in hp_cfg.items():
            if not is_spec(spec):
                continue

            path = f"{step_name}.hyperparams.{param_name}"
            ptype = spec["type"]

            # Categorical
            if ptype == "choice":
                choices = spec["choices"]
                space[path] = hp.choice(path, choices)

            # Integer
            elif ptype == "int":
                q = spec.get("step") if spec.get("step") is not None else 1
                space[path] = scope.int(hp.quniform(path, spec["min"], spec["max"], q))

            # Float
            elif ptype == "float":
                q = spec.get("step")
                if q is None:
                    space[path] = hp.uniform(path, spec["min"], spec["max"])
                else:
                    space[path] = hp.quniform(path, spec["min"], spec["max"], q)

            else:
                raise ValueError(f"Unsupported type '{ptype}' at {path}")

    return space


def combine_space_to_cfg(base_cfg: dict, sampled_params: dict) -> dict:
    """Build pipeline config by merging sampled hyperparameters with base config.

    The sampled_params dict contains only the tunable parameters that were sampled during optimization,
    with keys corresponding to their config paths (e.g. "model.hyperparams.max_depth").
    This function takes the base pipeline config and updates it with the sampled values at the correct paths,
    resulting in a complete config that can be used to instantiate and train the model.

    Parameters
    ----------
    base_cfg : dict
        The original pipeline config containing all components and hyperparameters,
            where tunable hyperparameters are defined as specs (type + range/choices).
    sampled_params : dict
        A dictionary of sampled hyperparameters from optimization, with keys as config paths.

    Returns
    -------
    dict
        A complete pipeline config with sampled hyperparameters applied.
    """
    cfg = copy.deepcopy(base_cfg)

    for path, value in sampled_params.items():
        keys = path.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    return cfg


def make_early_stop(patience: int, startup_jobs: int = 0) -> Callable:
    """Factory for early stopping function based on lack of improvement in loss.

    Early stopping is only evaluated after the startup phase (first `startup_jobs` trials)
    has completed. During the startup phase, all trials run unconditionally to allow
    TPE to gather enough data for model-guided search.

    Parameters
    ----------
    patience : int
        Number of consecutive trials without improvement in loss before stopping.
        The loss is expected to be a numeric value where lower is better.
    startup_jobs : int
        Number of initial random exploration trials to run before evaluating early stopping.
    """

    best_loss = float("inf")
    trials_without_improvement = 0

    def stop_fn(trials: Trials):
        nonlocal best_loss, trials_without_improvement

        if not trials.trials:
            return False, {}

        # Always run all startup jobs before evaluating early stopping
        if len(trials.trials) <= startup_jobs:
            current_loss = trials.trials[-1]["result"]["loss"]
            if current_loss < best_loss:
                best_loss = current_loss
            logger.info(
                f"Startup phase [{len(trials.trials)}/{startup_jobs}] — loss: {current_loss:.4f}"
            )
            return False, {}

        current_loss = trials.trials[-1]["result"]["loss"]

        if current_loss < best_loss:
            logger.info(f"Loss decreased from {best_loss:.4f} --> {current_loss:.4f}")
            best_loss = current_loss
            trials_without_improvement = 0
        else:
            trials_without_improvement += 1
            logger.info(
                f"No improvement in loss [{trials_without_improvement}/{patience}]"
            )

        if trials_without_improvement >= patience:
            logger.info(
                f"Early stopping triggered after {patience} trials without improvement."
            )
            return True, {}

        return False, {}

    return stop_fn


def objective(
    sampled_params, pipeline_cls: type[TuneableModel], data: pd.DataFrame, context: dict
) -> dict:
    """Objective function for hyperparameter optimization.

    This function will be called by hyperopt for each set of sampled hyperparameters.
    It combines the sampled hyperparameters with the base pipeline config, trains the model using k-fold cross-validation,
    evaluates the model using the specified metrics, and returns the loss value to be optimized
    along with any additional metrics and the config used for this trial.
    Function should be called with a pre-initialized partial that has the pipeline_cls, data, and context arguments set.

    Parameters
    ----------
    sampled_params : dict
        A dictionary of sampled hyperparameters from optimization, with keys as config paths.
    pipeline_cls : type[TuneableModel]
        The model class to instantiate and train, which must implement the TuneableModel protocol.
    data : pd.DataFrame
        The full dataset to use for training and evaluation (will be split into folds internally).
    context : dict
        The experiment context containing pipeline config, training settings, and optimization settings.

    Returns
    -------
    dict
        A dictionary containing the loss value for this trial, the status, any additional metrics collected during
        evaluation, and the config used for this trial.
    """

    # Combine base config with sampled hyperparameters to get the config for this trial
    cfg = combine_space_to_cfg(
        base_cfg=context["pipeline"], sampled_params=sampled_params
    )

    # Train and evaluate model
    evaluator = EvaluatorBuilder().from_dict(context["metrics"]).build()
    for fold, X_train, X_test, y_train, y_test in kfold_iterator(
        data, **context["training"]
    ):
        pipeline = pipeline_cls.from_config(cfg)
        pipeline.fit(X_train, y_train)
        evaluator.add_model_evaluation(model=pipeline, X=X_test, y_true=y_test)

    # The config specifies which metric to optimize as the "loss_metric".
    results = evaluator.get_results()
    key = context["optimization"]["loss_metric"]
    if key not in results["values"]:
        raise ValueError(
            f"Specified loss_metric '{key}' not found in evaluation results. Available metrics: {list(results['values'].keys())}"
        )
    loss_value = results["values"][key]
    loss = -loss_value if context["optimization"]["higher_is_better"] else loss_value

    return {
        "loss": loss,
        "status": STATUS_OK,
        "metrics": results,
        "cfg": cfg,
        "params": sampled_params,
    }


def optimize_model(
    pipeline_cls: type[TuneableModel],
    data: pd.DataFrame,
    context: dict,
) -> OptimizationResults:
    """Optimize model hyperparameters using hyperopt.

    Parameters
    ----------
    pipeline_cls : type[TuneableModel]
        The model class to optimize, which must implement the TuneableModel protocol.
    data : pd.DataFrame
        The full dataset to use for optimization (will be split into folds internally).
    context : Context
        The experiment context containing pipeline config and training settings.

    Returns
    -------
    OptimizationResults
        A dataclass containing the best hyperparameters found, the history of loss values across trials,
        and any additional metrics collected during optimization.
    """

    # Build search space from pipeline config
    search_space = build_search_space(context["pipeline"])

    # Set global random seed for reproducibility
    rstate = np.random.default_rng(42)

    # Make early stopping function with patience (only evaluated after startup phase)
    early_stop_fn = make_early_stop(
        patience=context["optimization"]["early_stop_patience"],
        startup_jobs=context["optimization"]["startup_jobs"],
    )

    # Preinit objective with data and context
    objective_fn = partial(
        objective,
        pipeline_cls=pipeline_cls,
        data=data,
        context=context,
    )

    # If no tunable hyperparameters, just run the objective once with the base config and return the results without optimization
    if not search_space:
        logger.warning(
            "No tunable hyperparameters found in config. Skipping optimization."
        )
        results = objective({}, pipeline_cls=pipeline_cls, data=data, context=context)
        final_pipeline = pipeline_cls.from_config(context["pipeline"])
        final_pipeline.fit(
            data.drop(columns=context["training"]["target_column"]),
            data[context["training"]["target_column"]],
        )
        return OptimizationResults(
            model=final_pipeline,
            best_params={},
            loss_history=[results["loss"]],
            metrics=results["metrics"],
        )

    # Run optimization
    trials = Trials()
    best = fmin(
        fn=objective_fn,
        space=search_space,
        algo=partial(
            tpe.suggest, n_startup_jobs=context["optimization"]["startup_jobs"]
        ),
        max_evals=context["optimization"]["max_evals"],
        trials=trials,
        rstate=rstate,
        early_stop_fn=early_stop_fn,
        verbose=False,
    )

    # Find best params and loss history
    loss_history = [t["result"]["loss"] for t in trials.trials]
    best = trials.best_trial
    logger.info(f"Optimization completed. Best loss: {best['result']['loss']:.4f}")
    logger.info(f"Best hyperparameters: {best['result']['params']}")

    # Train final model with best hyperparameters on full data
    final_pipeline = pipeline_cls.from_config(best["result"]["cfg"])
    final_pipeline.fit(
        data.drop(columns=context["training"]["target_column"]),
        data[context["training"]["target_column"]],
    )

    return OptimizationResults(
        model=final_pipeline,
        best_params=best["result"]["params"],
        loss_history=loss_history,
        metrics=best["result"]["metrics"],
    )
