from .pipeline import Pipeline
from .model.factory import model_factory
from .transformer.factory import transformer_factory

def pipeline_factory(model_config, transformer_config) -> Pipeline:
    """ Create a Pipeline based on the context configuration.

    Parameters
    ----------
    model_config : dict
        The model configuration containing model specifications.
        - name: str
            The name of the model to create.
        - hyperparams: dict
            A dictionary of hyperparameters to initialize the model.

    transformer_config : list of dict
        - name: str
            The name of the transformer to create.
        - hyperparams: dict
            A dictionary of hyperparameters to initialize the transformer.

    Returns
    -------
    Pipeline
        The constructed Pipeline object.
    """
    # Create transformers
    transformers = []
    for transformer in transformer_config:
        transformers.append(transformer_factory(transformer['name'], transformer['hyperparams']))

    # Create model
    model = model_factory(model_config['name'], model_config['hyperparams'])

    # Create and return the pipeline
    pipeline = Pipeline(transformers=transformers, model=model)
    return pipeline