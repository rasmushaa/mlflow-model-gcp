import yaml
import logging

logger = logging.getLogger(__name__)


class Context(dict):
    """ The main context for everything related to configuration and hyperparameters.

    Attributes
    ----------
    config: dict
        The model configuration loaded from config, and hyperparameters YAML files.
    config_flat: dict
        The flattened model configuration for easy logging with MLflow.
    """

    def __init__(self):
        """ Initialize the Context by loading configuration and hyperparameters from YAML files.
        
        The config and hyperparameters format is validated during loading.
        Actual values are not validated here.
        """
        # Initialize the parent dict class
        super().__init__()
        
        # Safely open and load the main configuration file
        config = self.__open_config("configs/config.yaml")

        # Safely open and load the model hyperparameters
        config['model']['hyperparams'] = self.__open_hyperparams(f"configs/model_hyperparams/{config['model']['hyperparams_file']}")

        # Safely open and load each transformer's hyperparameters
        for transformer in config['transformer']:
            transformer['hyperparams'] = self.__open_hyperparams(f"configs/transformer_hyperparams/{transformer['hyperparams_file']}")
        
        # Populate the dict with the config data
        self.update(config)
        logger.debug(f"Loaded configuration: {self}")

        
    def ravel(self):
        """ Get the full configuration as a flattened dictionary,
        using dot notation for nested keys.

        Returns
        -------
        config: dict
            The model configuration loaded from config.yaml
        """
        return self.__flatten_dict(dict(self))
    

    def __open_yaml(self, filepath: str) -> dict:
        """ Open and load a YAML file.

        Parameters
        ----------
        filepath: str
            The path to the YAML file.

        Returns
        -------
        dict
            The loaded YAML content as a dictionary.
        """
        try:
            with open(filepath, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file not found: {filepath}")
        

    def __open_config(self, filepath: str) -> dict:
        """ Open and load the main configuration YAML file.

        Parameters
        ----------
        filepath: str
            The path to the configuration YAML file.

        Returns
        -------
        dict
            The loaded configuration as a dictionary.
        """
        config = self.__open_yaml(filepath)
        self.__validate_config(config)
        return config
    
    
    def __validate_config(self, config: dict):
        """ Validate the loaded configuration.

        Raises
        ------
        ValueError
            If required configuration keys are missing or invalid.
        """
        if not isinstance(config, dict):
            raise ValueError(f"Context() Configuration must be a dict. Got {config} instead")
        
        required_keys = ['model', 'transformer', 'training', 'query']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Context() Missing required configuration key: {key} in config.yaml")
            
        if not isinstance(config['transformer'], list):
            raise ValueError("Context() 'transformer' value must be a list of transformer configs")
        

    def __open_hyperparams(self, filepath: str) -> dict:
        """ Open and load a hyperparameters YAML file.

        Parameters
        ----------
        filepath: str
            The path to the hyperparameters YAML file.

        Returns
        -------
        dict
            The loaded hyperparameters as a dictionary.
        """
        hyperparams = self.__open_yaml(filepath)
        self.__validate_hyperparams(hyperparams)
        return hyperparams
            

    def __validate_hyperparams(self, hyperparams: dict):
        """ Validate the hyperparameters dictionary.
        
        Parameters
        ----------
        hyperparams: dict
            The hyperparameters dictionary to validate

        Raises
        ------
        ValueError
            If hyperparameters are missing, not a dict, empty, or contain invalid keys/values
        """
        if not isinstance(hyperparams, dict):
            raise ValueError(f"Context() Hyperparameters must be a dict. Got {hyperparams} instead")

        if not hyperparams:
            raise ValueError("Context() Hyperparameters must not be empty")

        for key, value in hyperparams.items():
            if not isinstance(key, str):
                raise ValueError(f"Context() All hyperparameter keys must be strings. Got invalid key: {key}")


    def __flatten_dict(self, d: dict, parent_key: str = '') -> dict:
        """Recursively flatten a nested dictionary using dot notation for keys.

        Examples
        --------
        {'a': {'b': 1}} -> {'a.b': 1}
        {'a': [{'b': 1}, {'c': 2}]} -> {'a.0.b': 1, 'a.1.c': 2}

        Parameters
        ----------
        d: dict
            The dictionary to flatten
        parent_key: str
            The prefix for keys (used during recursion)

        Returns
        -------
        dict
            A new dictionary with flattened keys
        """
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else str(k)
            # Recurse into nested dictionaries
            if isinstance(v, dict):
                items.update(self.__flatten_dict(v, new_key))
            # Treat list indices as integer keys and keep parsing
            # Avoid flattening hyperparams lists, remove for general use
            elif isinstance(v, list) and 'hyperparams' not in parent_key:
                for i, elem in enumerate(v):
                    list_key = f"{new_key}.{i}"
                    if isinstance(elem, dict):
                        items.update(self.__flatten_dict(elem, list_key))
                    elif isinstance(elem, list):
                        nested_dict = {str(idx): val for idx, val in enumerate(elem)}
                        items.update(self.__flatten_dict(nested_dict, list_key))
                    else:
                        items[list_key] = elem
            else:
                items[new_key] = v
        return items