from preprocessors.custom.feature_selector import FeatureSelector
from preprocessors.custom.kbest_text_vector import KbestTextVector


def create_preprocessors(preprocessor_config: dict) -> dict:
    """ Create preprocessors based on the configuration.

    The function reads the 'preprocessor' section from the context configuration,
    initializes the specified preprocessor classes with their hyperparameters,
    and returns a dictionary of preprocessor instances.

    To add a new preprocessor, implement the class and add it to the options dictionary.
    If no match is found, a ValueError is raised, and the available options are listed.
    
    Parameters
    ----------
    preprocessor_config: dict
        The context configuration containing preprocessor specifications.

    Returns
    -------
    preprocessors: dict
        A dictionary of initialized preprocessor instances with keys as specified in the config.
    """

    options = {
        'feature_selector': FeatureSelector,
        'kbest_text_vector': KbestTextVector,
    }

    preprocessors = {}
    for key, preprocessor in preprocessor_config.items():
        name = preprocessor['name']
        if name in options:
            PreprocessorClass = options[name]
            preprocessors[key] = PreprocessorClass(**preprocessor['hyperparams'])
        else:
            raise ValueError(f'Unknown preprocessor type: {name}. Available options: {list(options.keys())}')

    return preprocessors