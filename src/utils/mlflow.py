import os
import shutil
import joblib
from datetime import datetime
import mlflow
import pandas as pd
import numpy as np
import sklearn
from utils.git import get_git_metadata
from polymodel.model.wrapper import ModelWrapper


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

    
    def log_metrics(self, metrics: dict, step: int = None):
        ''' Log a dictionary of metrics to the current MLflow run.

        Parameters
        ----------
        metrics: dict
            A dictionary of metrics to log
        step: int, optional
            The step at which to log the metrics
        '''
        mlflow.log_metrics(metrics, step=step)


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


    def log_figures(self, figures_dict: dict):
        ''' Log a matplotlib figure artifact to the current MLflow run.

        Parameters
        ----------
        figures_dict: dict
            A dictionary of figure name (str) to matplotlib.figure.Figure
        '''
        for name, fig in figures_dict.items():
            mlflow.log_figure(fig, f'plots/{name}.png')


    def log_model(self, model: sklearn.base.BaseEstimator, preprocessors: dict, input_example: pd.DataFrame = None):
        ''' Log a MLflow model artifact to the current MLflow run.

        Parameters
        ----------
        model: sklearn.base.BaseEstimator
            The trained model to log
        preprocessors: dict, optional
            A dictionary of preprocessing steps to log along with the model
        '''
        # Create artifacts directory if it doesn't exist
        os.makedirs('artifacts', exist_ok=True)
        
        # Save model to temporary file
        artifact_paths = {'model': 'artifacts/model.pkl'}
        joblib.dump(model, artifact_paths['model'])

        # Save preprocessors to temporary files
        for index, preprocessor in preprocessors.items():
            artifact_paths[f'preprocessor_{index}'] = f'artifacts/preprocessor_{index}.pkl'
            joblib.dump(preprocessor, artifact_paths[f'preprocessor_{index}'])

        wheel_src = "dist/polymodel-1.0.0-py3-none-any.whl"
        wheel_dst = f"artifacts/{os.path.basename(wheel_src)}"
        shutil.copy(wheel_src, wheel_dst)
        artifact_paths['wheel'] = wheel_dst

        # Log the model with artifacts (code-based model using wrapper.py)
        mlflow.pyfunc.log_model(
            name='model',
            python_model="polymodel/src/polymodel/model/wrapper.py",
            artifacts=artifact_paths,
            #code_paths=['src/model_src/model/wrapper.py'],
            input_example=input_example,
            pip_requirements=[f"polymodel @ file://{{ARTIFACT_PATH}}/{os.path.basename(wheel_src)}"],
        )
        
    def __find_or_create_experiment(self):
        ''' Find an existing MLflow experiment by name, 
        or create it if not found.
        
        Returns
        -------
        experiment: mlflow.entities.Experiment
            The found or newly created experiment object
        '''
        experiment = mlflow.get_experiment_by_name(name=self.__experiment_name)
        if not experiment:
            mlflow.create_experiment(name=self.__experiment_name)
            experiment = mlflow.get_experiment_by_name(name=self.__experiment_name)
        return experiment
    

    def __log_default_params(self):
        ''' Log default parameters to the new MLflow run.
        
        The default parameters include Git metadata for reproducibility.
        '''
        git_metadata = get_git_metadata()
        mlflow.set_tags(git_metadata)
    

    def __create_run_name(self):
        ''' Create a run name based on current datetime.
        
        Returns
        -------
        run_name: str
            The created run name string
        '''
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f'Run {time_str}'