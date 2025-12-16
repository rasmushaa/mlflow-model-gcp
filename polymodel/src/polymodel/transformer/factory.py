from typing import Protocol
from .feature_selector import FeatureSelector
from .kbest_text_vector import KbestTextVector


################################# Interfaces #################################
class BaseTransformer(Protocol):
    def fit(self, X, y=None) -> None: ...
    def transform(self, X): ...
    def fit_transform(self, X, y=None): ...
    @property
    def features(self) -> list: ...


################################ Factory Function #################################
def transformer_factory(name: str, hyperparams: dict) -> BaseTransformer:
    """ Create transformers based on the configuration.

    The factory function initializes and returns a transformer instance
    based on the provided name and hyperparameters.
    
    Parameters
    ----------
    name : str
        The name of the transformer to create.
    hyperparams : dict
        A dictionary of hyperparameters to initialize the transformer.

    Returns
    -------
    BaseTransformer
        An initialized transformer instance.
    """

    options = {
        'feature_selector': FeatureSelector,
        'kbest_text_vector': KbestTextVector,
    }

    if name not in options:
        raise ValueError(f'Unknown transformer type: {name}. Available options: {list(options.keys())}')

    return options[name](**hyperparams)