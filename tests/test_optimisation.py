from types import SimpleNamespace

import numpy as np
import pandas as pd

import src.optimise as optimise_module
from src.optimise import (
    build_search_space,
    combine_space_to_cfg,
    is_spec,
    make_early_stop,
)


def test_is_spec():

    cfg = {
        "param1": {"type": "int", "min": 1, "max": 10},
        "param2": {"type": "float", "min": 0.0, "max": 1.0},
        "param3": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.1},
        "param4": {"type": "choice", "choices": ["a", "b", "c"]},
        "param5": 5,
    }

    assert is_spec(cfg["param1"])
    assert is_spec(cfg["param2"])
    assert is_spec(cfg["param3"])
    assert is_spec(cfg["param4"])
    assert not is_spec(cfg["param5"])


def test_build_search_space():
    cfg = {
        "model1": {
            "hyperparams": {
                "max_depth": {"type": "int", "min": 1, "max": 10},
                "learning_rate": {
                    "type": "float",
                    "min": 0.01,
                    "max": 0.1,
                    "step": 0.01,
                },
                "subsample": False,
            }
        },
        "model2": {
            "hyperparams": {
                "n_estimators": 15,
                "subsample": True,
                "criterion": {"type": "choice", "choices": ["gini", "entropy"]},
            }
        },
    }

    space = build_search_space(cfg)
    print(space)

    assert space["model1.hyperparams.max_depth"].name == "int"
    assert space["model1.hyperparams.learning_rate"].name == "float"
    assert space["model2.hyperparams.criterion"].name == "switch"


def test_combine_space_to_cfg():
    base_cfg = {
        "model1": {
            "hyperparams": {
                "max_depth": {"type": "int", "min": 1, "max": 10},
                "learning_rate": {
                    "type": "float",
                    "min": 0.01,
                    "max": 0.1,
                    "step": 0.01,
                },
                "subsample": False,
            }
        },
        "model2": {
            "hyperparams": {
                "n_estimators": 15,
                "subsample": True,
                "criterion": {"type": "choice", "choices": ["gini", "entropy"]},
            }
        },
    }

    sampled_params = {
        "model1.hyperparams.max_depth": 5,
        "model1.hyperparams.learning_rate": 0.05,
        "model2.hyperparams.criterion": "entropy",
    }

    cfg = combine_space_to_cfg(base_cfg, sampled_params)
    print(cfg)

    assert cfg["model1"]["hyperparams"]["max_depth"] == 5
    assert cfg["model1"]["hyperparams"]["learning_rate"] == 0.05
    assert cfg["model1"]["hyperparams"]["subsample"] == False  # unchanged
    assert cfg["model2"]["hyperparams"]["n_estimators"] == 15  # unchanged
    assert cfg["model2"]["hyperparams"]["subsample"] == True  # unchanged
    assert cfg["model2"]["hyperparams"]["criterion"] == "entropy"


def test_make_early_stop():

    def mk_trials(losses):
        """Create a minimal stand-in for hyperopt.Trials."""
        return SimpleNamespace(trials=[{"result": {"loss": l}} for l in losses])

    # Test triggering early stop after 3 non-improving trials (no startup phase)
    stop = make_early_stop(patience=2)
    assert stop(mk_trials([10.0])) == (False, {})
    assert stop(mk_trials([10.0, 11.0])) == (False, {})
    assert stop(mk_trials([10.0, 11.0, 12.0])) == (True, {})

    # Test reset of best loss after improvement (no startup phase)
    stop = make_early_stop(patience=2)
    assert stop(mk_trials([10.0])) == (False, {})
    assert stop(mk_trials([10.0, 9.0])) == (False, {})
    assert stop(mk_trials([10.0, 9.0, 9.5])) == (False, {})
    assert stop(mk_trials([10.0, 9.0, 9.5, 10.0])) == (
        True,
        {},
    )  # no improvement for 2 trials

    # Test startup phase: early stop is never triggered during startup_jobs
    stop = make_early_stop(patience=1, startup_jobs=3)
    assert stop(mk_trials([10.0])) == (False, {})  # startup 1/3
    assert stop(mk_trials([10.0, 20.0])) == (False, {})  # startup 2/3
    assert stop(mk_trials([10.0, 20.0, 30.0])) == (
        False,
        {},
    )  # startup 3/3, still no early stop
    assert stop(mk_trials([10.0, 20.0, 30.0, 40.0])) == (
        True,
        {},
    )  # past startup, patience=1 exceeded

    # Test startup phase tracks best loss so improvement resets after startup
    stop = make_early_stop(patience=1, startup_jobs=2)
    assert stop(mk_trials([10.0])) == (False, {})  # startup 1/2, best=10
    assert stop(mk_trials([10.0, 5.0])) == (False, {})  # startup 2/2, best=5
    assert stop(mk_trials([10.0, 5.0, 4.0])) == (
        False,
        {},
    )  # post-startup, improvement (4 < 5)
    assert stop(mk_trials([10.0, 5.0, 4.0, 4.5])) == (
        True,
        {},
    )  # no improvement, patience=1


def test_optimize_model_happy_path(monkeypatch):

    # Mock context and data
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    context = {
        "pipeline": {"some": "cfg"},
        "training": {"target_column": "y"},
        "optimization": {"early_stop_patience": 2, "max_evals": 5, "startup_jobs": 0},
    }

    # Mock pipeline class
    class DummyPipeline:
        def __init__(self, cfg):
            self.cfg = cfg
            self.fitted = False

        @classmethod
        def from_config(cls, cfg):
            return cls(cfg)

        def fit(self, X, y):
            self.fitted = True
            self.fit_X = X
            self.fit_y = y
            return self

    # Mock used functions and classes in optimise_module
    def fake_build_search_space(pipeline_cfg):
        assert pipeline_cfg == context["pipeline"]
        return {"space": "ok"}

    class FakeTrials:
        def __init__(self):
            self.trials = []

        @property
        def best_trial(self):
            # choose lowest loss
            return min(self.trials, key=lambda t: t["result"]["loss"])

    def fake_fmin(fn, space, algo, max_evals, trials, rstate, early_stop_fn, verbose):
        assert space == {"space": "ok"}
        assert max_evals == context["optimization"]["max_evals"]
        assert verbose is False

        # simulate hyperopt calling objective and recording results:
        trials.trials.append(
            {
                "result": {
                    "loss": 0.9,
                    "params": {"a": 1},
                    "cfg": {"pipeline": {"p": 1}},
                    "metrics": {"m": 1},
                }
            }
        )
        trials.trials.append(
            {
                "result": {
                    "loss": 0.7,
                    "params": {"a": 2},
                    "cfg": {"pipeline": {"p": 2}},
                    "metrics": {"m": 2},
                }
            }
        )
        trials.trials.append(
            {
                "result": {
                    "loss": 0.8,
                    "params": {"a": 3},
                    "cfg": {"pipeline": {"p": 3}},
                    "metrics": {"m": 3},
                }
            }
        )
        return {"a": 2}

    monkeypatch.setattr(optimise_module, "build_search_space", fake_build_search_space)
    monkeypatch.setattr(optimise_module, "Trials", FakeTrials)
    monkeypatch.setattr(
        optimise_module,
        "make_early_stop",
        lambda patience, startup_jobs=0: (lambda trials: (False, {})),
    )
    monkeypatch.setattr(
        np.random, "default_rng", lambda seed: SimpleNamespace(seed=seed)
    )
    monkeypatch.setattr(optimise_module, "fmin", fake_fmin)

    # Run optimization
    res = optimise_module.optimize_model(DummyPipeline, df, context)

    assert res.best_params == {"a": 2}
    assert res.loss_history == [0.9, 0.7, 0.8]
    assert res.metrics == {"m": 2}
    assert isinstance(res.model, DummyPipeline)
    assert res.model.fitted is True
