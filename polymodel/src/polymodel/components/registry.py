from .models.naive_bayes import NaiveBayesModel
from .models.random_forest import RandomForestModel
from .transformers.text_cleaner import TextCleaner
from .transformers.text_vectorizer import TextVevtorizer

# Registry of available components
COMPONENTS = {
    "random_forest": RandomForestModel,
    "naive_bayes": NaiveBayesModel,
    "text_cleaner": TextCleaner,
    "text_vectorizer": TextVevtorizer,
}


def get_component(name: str):
    """Retrieve a component class by name from the registry.

    Parameters
    ----------
    name : str
        The name of the component to retrieve.

    Returns
    -------
    class
        The component class corresponding to the given name.

    Raises
    ------
    ValueError
        If the specified component name is not found in the registry.
    """
    if name not in COMPONENTS:
        raise ValueError(
            f"Component '{name}' not found in registry.\nAvailable components: {list(COMPONENTS.keys())}"
        )
    return COMPONENTS[name]
