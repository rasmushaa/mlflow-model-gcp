from .feature_selector import FeatureSelector
from .interface import TransformerInterface
from .kbest_text_vector import KbestTextVector
from .text_cleaner import TextCleaner


def transformer_factory(name: str, hyperparams: dict) -> TransformerInterface:
    """Create transformers based on the configuration.

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
        "feature_selector": FeatureSelector,
        "kbest_text_vector": KbestTextVector,
        "text_cleaner": TextCleaner,
    }

    if name not in options:
        raise ValueError(
            f"Unknown transformer type: {name}. Available options: {list(options.keys())}"
        )

    return options[name](**hyperparams)
