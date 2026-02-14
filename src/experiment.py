import logging
import os
import shutil
from datetime import datetime
from importlib.metadata import version
from typing import Optional

import joblib
import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentManager:
    """A manager class to handle MLflow experiments and runs.

    Pulls the experiment settings from environment variables,
    creates new experiment with given name if not found,
    and automatically logs default parameters to new runs.

    It is possible to resume an existing run within the experiment,
    by calling start_run() multiple times on the same instance.

    A single ExperimentManager instance manages one experiment.
    To create a new experiment, create a new instance.
    """

    def __init__(self):
        self.__experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "default")
        self.__experiment = self.__find_or_create_experiment()
        self.__run = None
        logger.debug(
            f"Initialized ExperimentManager for experiment: {self.__experiment_name}"
        )

    def start_run(self) -> mlflow.entities.Run:
        """Start an MLflow run within the managed experiment.

        On first call, creates a new run, and autolog default parameters.
        On subsequent calls, resumes the existing run to append data.
        To create a new run, create a new ExperimentManager instance.

        Returns
        -------
        run: mlflow.entities.Run
            The started or resumed MLflow run object
        """
        if self.__run is None:
            self.__run = mlflow.start_run(
                experiment_id=self.__experiment.experiment_id,
                run_name=self.__create_run_name(),
            )
            self.__log_default_params()
        else:
            self.__run = mlflow.start_run(run_id=self.__run.info.run_id)

        return self.__run

    def end_run(self):
        """End the current MLflow run if active.

        The recommended run method is to use a context manager,
        but if not, this method provides an intuitive way to end the run.
        """
        if self.__run is not None:
            mlflow.end_run()
            self.__run = None

    def log_params(self, params: dict):
        """Log a dictionary of parameters to the current MLflow run.

        Parameters
        ----------
        params: dict
            A dictionary of parameters to log
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: Optional[int] = None, decimals: int = 3):
        """Log a dictionary of metrics to the current MLflow run.

        Forces unique timestamps for each metric to avoid overwriting.
        The model logging in MLflow copys the metrics from the current run,
        and but the database does not suport duplicated rows.

        Example error:
        DETAIL:  Key (key, "timestamp", step, run_uuid, value, is_nan)=
        (roc_auc.CLOTHING.mean, 1765208298936, 0, abe53812940349ec8077a251eaa92a3b, 0, t) already exists.

        Parameters
        ----------
        metrics: dict
            A dictionary of metrics to log
        step: int, optional
            The mlflow step at which to log the metrics
        decimals: int, optional
            The number of decimal places to round the metric values
        """
        for i, (k, v) in enumerate(metrics.items()):
            mlflow.log_metric(k, round(v, decimals), step=step)

    def log_metric(
        self, key: str, value: float, step: Optional[int] = None, decimals: int = 3
    ):
        """Log a single metric to the current MLflow run.

        Parameters
        ----------
        key: str
            The name of the metric to log
        value: float
            The value of the metric to log
        step: int, optional
            The mlflow step at which to log the metric
        decimals: int, optional
            The number of decimal places to round the metric value
        """
        mlflow.log_metric(key, round(value, decimals), step=step)

    def log_input(self, df, name: str):
        """Log an array-like input data artifact to the current MLflow run.

        Parameters
        ----------
        df: pd.DataFrame or np.ndarray
            The input data to log
        name: str
            The name of the logged data artifact
        """
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if isinstance(df, pd.DataFrame):
            df = mlflow.data.from_pandas(df)
        if isinstance(df, np.ndarray) or isinstance(df, list):
            df = mlflow.data.from_numpy(df)
        mlflow.log_input(df, name)

    def log_text(self, text: str, name: str):
        """Log a text artifact to the current MLflow run.

        Parameters
        ----------
        text: str
            The text content to log
        name: str
            The name of the logged text artifact
        """
        mlflow.log_text(text, f"{name}.txt")

    def log_dict(self, data_dict: dict, name: str):
        """Log a dictionary artifact to the current MLflow run.

        Parameters
        ----------
        data_dict: dict
            The dictionary to log
        name: str
            The name of the logged dictionary artifact
        """
        mlflow.log_dict(data_dict, f"{name}.json")

    def log_figures(self, figures_dict: dict):
        """Log a matplotlib figure artifact to the current MLflow run.

        Parameters
        ----------
        figures_dict: dict
            A dictionary of figure name (str) to matplotlib.figure.Figure
        """
        for name, fig in figures_dict.items():
            mlflow.log_figure(fig, f"{name}.png")

    def set_tags(self, tags: dict):
        """Set tags to the current MLflow run.

        Parameters
        ----------
        tags: dict
            A dictionary of tags to set
        """
        mlflow.set_tags(tags)

    def log_model(self, pipeline, data_example: pd.DataFrame):
        """Log a MLflow model artifact to the current MLflow run.

        Parameters
        ----------
        pipeline:
            The trained model class to log
        data_example: pd.DataFrame
            A sample DataFrame of any size to create input example from.
            Note: only top 5 rows are used, and only the features used by the pipeline.
        """
        # The registered model name matches the runtime env (e.g., BankingModel-dev)
        model_name = os.getenv("MLFLOW_MODEL_NAME")
        assert model_name, "MLFLOW_MODEL_NAME environment variable is not set."

        # Save model artifacts to local temp directory
        artifact_paths = self.__save_model_artifacts_to_local(pipeline)

        # Prepare input example
        input_example = self.__create_model_input_example(data_example, pipeline)

        # Log model metadata
        self.log_dict(pipeline.layers, "layers")
        self.set_tags({"model.architecture": pipeline.architecture})
        self.set_tags({"model.features": pipeline.features})

        # Log polymodel version constraints
        current_version = version("polymodel")
        logger.info(
            f"Logging model with polymodel version constraint: ~= {current_version}"
        )

        # Log the model with artifacts (code-based model using wrapper.py)
        logged_model = mlflow.pyfunc.log_model(
            name="LoggedModel",
            python_model="polymodel/src/polymodel/wrapper.py",
            artifacts=artifact_paths,
            input_example=input_example,
            pip_requirements=[
                f"polymodel~={current_version}",
            ],
        )

        # Register the model to the MLflow Model Registry. Todo: alias the returned version
        _ = mlflow.register_model(
            f"models:/{logged_model.model_uri.split('/')[-1]}",
            model_name,
            tags={
                "package.version": current_version,
                "commit.sha": os.getenv("GIT_COMMIT_SHA", "unknown"),
                "model.features": pipeline.features,
                "model.architecture": pipeline.architecture,
            },
        )

    def __save_model_artifacts_to_local(self, pipeline) -> dict[str, str]:
        """Save model artifacts to local 'artifacts/' directory.

        The 'artifacts/' directory is created or overwritten.

        Parameters
        ----------
        pipeline:
            The trained model pipeline to save

        Returns
        -------
        artifact_paths: dict
            A dictionary with paths to the saved artifacts
        """
        artifact_paths = {
            "pipeline": "artifacts/pipeline.pkl",
        }
        if os.path.exists("artifacts"):
            shutil.rmtree("artifacts")
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(pipeline, artifact_paths["pipeline"])
        return artifact_paths

    def __create_model_input_example(
        self, data_example: pd.DataFrame, pipeline
    ) -> pd.DataFrame:
        """Create a small model input example DataFrame from the actual training data.

        Parameters
        ----------
        data_example: pd.DataFrame
            A small sample DataFrame to extract features from
        pipeline:
            The trained model pipeline containing feature information

        Returns
        -------
        input_example: pd.DataFrame
            The DataFrame containing only the features used by the model
        """
        input_example = data_example.iloc[:5]
        input_example = input_example[pipeline.features]  # Use only required features
        return input_example

    def __find_or_create_experiment(self):
        """Find an existing MLflow experiment by name,
        or create it if not found.

        Returns
        -------
        experiment: mlflow.entities.Experiment
            The found or newly created experiment object
        """
        experiment = mlflow.get_experiment_by_name(name=self.__experiment_name)

        # If experiment exists but is deleted, it should be restored or permanently deleted first
        if experiment and experiment.lifecycle_stage == "deleted":
            raise RuntimeError(
                f"Experiment '{self.__experiment_name}' exists in 'deleted' state. "
                "Run `mlflow experiments restore` or `mlflow gc`."
            )

        if not experiment:
            mlflow.create_experiment(name=self.__experiment_name)
            experiment = mlflow.get_experiment_by_name(name=self.__experiment_name)
        return experiment

    def __log_default_params(self):
        """Log default parameters to the new MLflow run.

        The default parameters include Git metadata for reproducibility,
        and model package version.
        """

        mlflow.set_tags(
            {
                "package.version": version("polymodel"),
                "commit.sha": os.getenv("GIT_COMMIT_SHA", "unknown"),
                "commit.user": os.getenv("GIT_COMMIT_USERNAME", "unknown"),
                "commit.author": os.getenv("GIT_COMMIT_AUTHOR_NAME", "unknown"),
                "commit.branch": os.getenv("GIT_COMMIT_BRANCH", "unknown"),
            }
        )

    def __create_run_name(self):
        """Create a run name based on current datetime.

        Returns
        -------
        run_name: str
            The created run name string
        """
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Run {time_str}"
