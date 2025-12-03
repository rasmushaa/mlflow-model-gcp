import pandas as pd
import mlflow.pyfunc
import mlflow.models
import joblib


class ModelWrapper(mlflow.pyfunc.PythonModel):
    """ MLflow model wrapper to save and load model with preprocessors. 
    
    This is the main class used by MLflow to save and load the model along with its
    preprocessing steps. It applies the preprocessors in sequence before making predictions
    with the trained model.
    """

    def load_context(self, context):
        """ Load model and preprocessors from artifacts.
        
        Inputs:
        context: dict
            MLflow model context with artifacts paths.
        """
        # Load preprocessors dynamically
        self.__preprocessors = {}
        for key, path in context.artifacts.items():
            if key.startswith("preprocessor_"):
                idx = key.replace("preprocessor_", "")
                self.__preprocessors[idx] = joblib.load(path)

        # Load the main model
        self.__model = joblib.load(context.artifacts['model'])


    def predict(self, context, model_input: pd.DataFrame):
        """ Apply preprocessors and model to input data.
        
        Inputs:
        context: dict
            MLflow model context with artifacts paths.
        model_input: pd.DataFrame
            Input data for prediction.
        
        Returns:
        pd.DataFrame or np.ndarray
            Model predictions.
        """
        # Copy input data to avoid modifying original
        X = model_input.copy()

        # Apply preprocessors in order
        for idx in sorted(self.__preprocessors.keys(), key=int):
            X = self.__preprocessors[idx].transform(X)

        # Make predictions with the trained model
        return self.__model.predict(X)


# Set this as the model for MLflow when loaded as code
mlflow.models.set_model(ModelWrapper())