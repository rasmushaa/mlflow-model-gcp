import os
import shutil
import joblib
from datetime import datetime
import mlflow
import pandas as pd
import numpy as np
import logging
from .package import get_current_version, get_hash_of_file

logger = logging.getLogger(__name__)


class ExperimentManager:
    ''' A manager class to handle MLflow experiments and runs.
    
    Pulls the experiment settings from environment variables,
    creates new experiment with given name if not found,
    and automatically logs default parameters to new runs.

    It is possible to resume an existing run within the experiment,
    by calling start_run() multiple times on the same instance.

    A single ExperimentManager instance manages one experiment.
    To create a new experiment, create a new instance.
    '''
    def __init__(self):
        self.__experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'default')
        self.__experiment = self.__find_or_create_experiment()
        self.__run = None
        logger.debug(f"Initialized ExperimentManager for experiment: {self.__experiment_name}")


    def start_run(self) -> mlflow.entities.Run:
        ''' Start an MLflow run within the managed experiment.

        On first call, creates a new run, and autolog default parameters.
        On subsequent calls, resumes the existing run to append data.
        To create a new run, create a new ExperimentManager instance.

        Returns
        -------
        run: mlflow.entities.Run
            The started or resumed MLflow run object
        '''
        if self.__run is None:
            self.__run = mlflow.start_run(
                experiment_id=self.__experiment.experiment_id, 
                run_name=self.__create_run_name()
                )
            self.__log_default_params()
        else:
            self.__run = mlflow.start_run(run_id=self.__run.info.run_id)
            
        return self.__run 
    

    def end_run(self):
        ''' End the current MLflow run if active.

        The recommended run method is to use a context manager,
        but if not, this method provides an intuitive way to end the run.
        '''
        if self.__run is not None:
            mlflow.end_run()
            self.__run = None


    def log_params(self, params: dict):
        ''' Log a dictionary of parameters to the current MLflow run.

        Parameters
        ----------
        params: dict
            A dictionary of parameters to log
        '''
        mlflow.log_params(params)

    
    def log_metrics(self, metrics: dict, step: int = None, decimals: int = 3):
        ''' Log a dictionary of metrics to the current MLflow run.

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
        '''
        for i, (k, v) in enumerate(metrics.items()):
            mlflow.log_metric(k, round(v, decimals), step=step)



    def log_input(self, df, name: str):
        ''' Log an array-like input data artifact to the current MLflow run.

        Parameters
        ----------
        df: pd.DataFrame or np.ndarray
            The input data to log
        name: str
            The name of the logged data artifact
        '''
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if isinstance(df, pd.DataFrame):
            df = mlflow.data.from_pandas(df)
        if isinstance(df, np.ndarray) or isinstance(df, list):
            df = mlflow.data.from_numpy(df)
        mlflow.log_input(df, name)


    def log_text(self, text: str, name: str):
        ''' Log a text artifact to the current MLflow run.

        Parameters
        ----------
        text: str
            The text content to log
        name: str
            The name of the logged text artifact
        '''
        mlflow.log_text(text, f'text/{name}.txt')


    def log_dict(self, data_dict: dict, name: str):
        ''' Log a dictionary artifact to the current MLflow run.

        Parameters
        ----------
        data_dict: dict
            The dictionary to log
        name: str
            The name of the logged dictionary artifact
        '''
        mlflow.log_dict(data_dict, f'dict/{name}.json')


    def log_figures(self, figures_dict: dict):
        ''' Log a matplotlib figure artifact to the current MLflow run.

        Parameters
        ----------
        figures_dict: dict
            A dictionary of figure name (str) to matplotlib.figure.Figure
        '''
        for name, fig in figures_dict.items():
            mlflow.log_figure(fig, f'plot/{name}.png')


    def log_model(self, pipeline, data_example: pd.DataFrame, wheel_path: str):
        ''' Log a MLflow model artifact to the current MLflow run.

        Parameters
        ----------
        pipeline:
            The trained model to log
        data_example: pd.DataFrame
            A small sample is extracted to use as input example mathinc the first layer features
        wheel_path: str
            The path to the model package wheel file
        '''
        MODEL_NAME = 'TransactionModel'

        # Create artifacts directory if it doesn't exist
        os.makedirs('artifacts', exist_ok=True)
        
        # Save model to temporary file
        artifact_paths = {'pipeline': 'artifacts/pipeline.pkl'}
        joblib.dump(pipeline, artifact_paths['pipeline'])

        # Add model package wheel as artifact
        wheel_dst = f"artifacts/{os.path.basename(wheel_path)}"
        shutil.copy(wheel_path, wheel_dst)
        artifact_paths['wheel'] = wheel_dst

        # Prepare input example
        input_example = data_example.iloc[:5]
        input_example = input_example[pipeline.features[0]['features']] # In case of extra RawFeatures

        # Prepare pip requirements with explicit versions
        pip_reqs = [
            f"polymodel @ file://{{ARTIFACT_PATH}}/{os.path.basename(wheel_path)}",
        ]

        # Log the model with artifacts (code-based model using wrapper.py)
        logged_model = mlflow.pyfunc.log_model(
            name='LoggedModel',
            python_model="polymodel/src/polymodel/wrapper.py",
            artifacts=artifact_paths,
            input_example=input_example,
            pip_requirements=pip_reqs,
        )

        # Register the model to the MLflow Model Registry
        registered_model = mlflow.register_model(
            f"models:/{logged_model.model_uri.split('/')[-1]}",
            MODEL_NAME
        )
        
        # Set an alias for the registered model version
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="challenger",
            version=registered_model.version
        )

        # Clean up temporary artifacts directory
        shutil.rmtree('artifacts')
        

    def __find_or_create_experiment(self):
        ''' Find an existing MLflow experiment by name, 
        or create it if not found.
        
        Returns
        -------
        experiment: mlflow.entities.Experiment
            The found or newly created experiment object
        '''
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
        ''' Log default parameters to the new MLflow run.
        
        The default parameters include Git metadata for reproducibility,
        and model package version and hash.
        '''
        version = get_current_version()
        file_hash = get_hash_of_file()
        git_sha = os.getenv("GIT_COMMIT_SHA")

        mlflow.set_tags({
            "model.package.version": version,
            "model.package.hash": file_hash,
            "git.commit.sha": git_sha
        })


    def __create_run_name(self):
        ''' Create a run name based on current datetime.
        
        Returns
        -------
        run_name: str
            The created run name string
        '''
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f'Run {time_str}'