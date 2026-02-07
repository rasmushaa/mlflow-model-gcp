import logging
from typing import List

import yaml

logger = logging.getLogger(__name__)


class Context(dict):
    """The main context for everything related to configuration and hyperparameters.

    The context class inherits from dict and is initialized by loading the configuration from a YAML file.
    It acts like a dictionary, but also provides a method to flatten the configuration for mlflow logging.

    Methods
    -------
    ravel()
        Get the full configuration as a flattened dictionary to log to mlflow, etc.
    """

    def __init__(self):
        """Initialize the Context by loading configuration and hyperparameters from YAML files.

        The config and hyperparameters format is validated during loading.
        Actual values are not validated here.
        """
        super().__init__()
        config = self.__open_config("config.yaml")
        self.update(config)
        logger.info(f"Loaded configuration: {self}")

    def ravel(self, exclude_keys: List[str] = []) -> dict:
        """Get the full configuration as a flattened dictionary

        Parameters
        ----------
        exclude_keys: List[str], optional
            List of top-level keys to exclude from flattening.
            This is useful metrics or other nested config sections
            that should not be flattened for mlflow logging.

        Returns
        -------
        config: dict
            The model configuration loaded from config.yaml
        """
        return self.__flatten_dict(dict(self), exclude_keys=exclude_keys)

    def __open_yaml(self, filepath: str) -> dict:
        """Open and load a YAML file.

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
            with open(filepath, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file not found: {filepath}")

    def __open_config(self, filepath: str) -> dict:
        """Open and load the main configuration YAML file.

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
        """Validate the loaded configuration.

        Raises
        ------
        ValueError
            If required configuration keys are missing or invalid.
        """
        if not isinstance(config, dict):
            raise ValueError(
                f"Context() Configuration must be a dict. Got {config} instead"
            )

        required_keys = ["model", "transformer", "training", "query"]
        for key in required_keys:
            if key not in config:
                raise ValueError(
                    f"Context() Missing required configuration key: {key} in config.yaml"
                )

        if not isinstance(config["transformer"], list):
            raise ValueError(
                "Context() 'transformer' value must be a list of transformer configs"
            )

    def __flatten_dict(self, d: dict, exclude_keys: List[str] = []) -> dict:
        """Flatten a nested dictionary using dot notation for keys.

        Parameters
        ----------
        d: dict
            The nested dictionary to flatten.
        exclude_keys: List[str], optional
            List of keys to exclude from flattening.

        Returns
        -------
        dict
            The flattened dictionary for mlflow logging.
        """

        items = {}
        for k, v in d.items():

            if k in exclude_keys:
                continue

            if k == "model":
                name = v.get("name", "unknown_model")
                hyperparams = v.get("hyperparams", {})
                for hk, hv in hyperparams.items():
                    items[f"model.{name}.{hk}"] = hv

            elif k == "transformer":
                for i, transformer in enumerate(v):
                    t_name = transformer.get("name", f"unknown_transformer_{i}")
                    t_hyperparams = transformer.get("hyperparams", {})
                    for tk, tv in t_hyperparams.items():
                        items[f"transformer.{t_name}.{tk}"] = tv

            elif isinstance(v, dict):
                sub_items = self.__flatten_dict(v)
                for sub_k, sub_v in sub_items.items():
                    items[f"{k}.{sub_k}"] = sub_v

            else:
                items[k] = v

        return items
