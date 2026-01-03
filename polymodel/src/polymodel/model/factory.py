from .interface import ModelInterface
from .random_forest import RandomForestModel


def model_factory(name: str, hyperparams: dict) -> ModelInterface:
    """Factory function to create machine learning models.

    To add a new model, inherit from ModelInterface
    and add it to the options dictionary.

    Parameters
    ----------
    name: str
        The name of the model to create.
    hyperparams: dict
        A dictionary of hyperparameters to initialize the model.

    Returns
    -------
    model: sklearn.base.BaseEstimator
        An instance of the specified machine learning model.
    """

    options = {
        "random_forest": RandomForestModel,
    }

    if name not in options:
        raise ValueError(
            f"create_model() Model '{name}' is not supported. Available models: {list(options.keys())}"
        )

    return options[name](**hyperparams)
