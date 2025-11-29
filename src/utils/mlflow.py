import os
from datetime import datetime
import mlflow
import pandas as pd
import numpy as np
from utils.git import get_git_metadata


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