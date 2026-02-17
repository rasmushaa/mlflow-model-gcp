import logging
from typing import Any, cast

from .components.base_components import BaseComponent, BaseModel
from .components.registry import get_component

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, components: list[BaseComponent]) -> None:
        """Init the pipeline with a list of components

        The last component must be a model (i.e., inherit from BaseModel),
        and the rest can be any transformers or models.

        Parameters
        ----------
        components : list[BaseComponent]
            A list of BaseComponent instances that make up the pipeline.

        Raises
        ------
        ValueError
            If the components list is empty or if the last component does not inherit from BaseModel.
        """
        if not components:
            raise ValueError("Pipeline must contain at least one component.")
        if not isinstance(components[-1], BaseModel):
            raise ValueError(
                "Pipeline validation failed: last component must inherit from BaseModel."
            )
        self._components = components
        logger.debug(self.__str__())

    def __str__(self) -> str:
        """String representation of the pipeline architecture."""
        return f"Pipeline(architecture={self.architecture})"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Pipeline":
        """Init a Pipeline instance from a configuration dictionary.

        Example
        -------
        ```python
        config = {
            "scaler": {
                "features": ["feature1", "feature2"],
                "transform_mode": "overwrite",
                "transform_suffix": "_scaled",
                "hyperparams": {
                    "method": "standard"
                }
            },
            "model": {
                "features": ["*"],
                "transform_mode": "overwrite",
                "transform_suffix": "",
                "hyperparams": {} # Optional defaults to sklearn
            }
        }
        pipeline = Pipeline.from_config(config)
        ```

        Parameters
        ----------
        config : dict[str, Any]
            A configuration dictionary. The same pattern is used for each component.

        Returns
        -------
        Pipeline
            An instance of the Pipeline class initialized according to the provided configuration.
        """

        steps: list[tuple[str, dict[str, Any]]] = []
        if isinstance(config, dict):
            for name, step_cfg in config.items():
                if not isinstance(step_cfg, dict):
                    raise TypeError(f"Pipeline component `{name}` must be a mapping.")
                steps.append((name, step_cfg))
        else:
            raise TypeError("`pipeline` must be a mapping.")

        def build_component(name: str, step_cfg: dict[str, Any]) -> BaseComponent:
            component_cls = get_component(name)
            try:
                hyperparams = (
                    step_cfg.get("hyperparams", {})
                    if step_cfg.get("hyperparams", {}) is not None
                    else {}
                )
                return component_cls(
                    features=step_cfg["features"],
                    transform_mode=step_cfg["transform_mode"],
                    transform_suffix=step_cfg["transform_suffix"],
                    **hyperparams,
                )
            except KeyError as e:
                raise KeyError(
                    f"Missing required key `{e.args[0]}` in pipeline component `{name}` configuration."
                )

        components = [build_component(name, step_cfg) for name, step_cfg in steps]
        return cls(components)

    @property
    def classes(self) -> list[str]:
        """Get the classes from the last model component of the pipeline.

        Returns
        -------
        list[str]
            A list of class labels from the last model component.
        """
        return cast(BaseModel, self._components[-1]).classes

    @property
    def architecture(self) -> str:
        """Get a string representation of the pipeline architecture.

        Returns
        -------
        str
            A string describing the sequence of components in the pipeline.
        """
        return "->".join([c.__class__.__name__ for c in self._components])

    @property
    def layers(self) -> dict:
        """Get the signatures of the components in the pipeline.

        Returns
        -------
        dict
            A dictionary containing the names and features of each component in the pipeline.
        """
        signatures = {}
        for i, component in enumerate(self._components):
            entry = {
                "name": component.__class__.__name__,
                "signature": component.signature,
                "resolved_features": component.resolved_features,
            }
            signatures[i] = entry
        return signatures

    @property
    def resolved_features(self) -> list[str]:
        """Get the list of actually required features by the pipeline.

        Depending on the architecture of the pipeline, some features might be
        transformed or dropped by the transformers. This property returns the final
        list of mandatory features required by the pipeline.

        Details
        -------
        The features are determined by checking which first component signature values
        are used in at least one layer of the pipeline.
        Note, some layers may create new features that are not part of the input signature.

        Returns
        -------
        List[str]
            A list of feature names that are required by the pipeline.
        """
        pipeline_signature = self._components[0].signature
        required_features = set()
        for component in self._components:
            component_features = component.resolved_features
            for feature in pipeline_signature:
                if feature in component_features:
                    required_features.add(feature)
        return list(required_features)

    def fit(self, X, y=None):
        """Fit the pipeline to the data.

        Parameters
        ----------
        X : array-like
            The input features for training.
        y : array-like, optional
            The target labels for training (required if the last component is a supervised model).

        Returns
        -------
        self
            The fitted pipeline instance.
        """
        logger.info("Fiting pipeline...")
        for component in self._components:
            X = component.fit(X, y).transform(X)
        logger.debug("Resolved layers:\n%s", self.layers)
        return self

    def predict(self, X):
        """Make predictions using the fitted pipeline.

        Parameters
        ----------
        X : array-like
            The input features for prediction.

        Returns
        -------
        array-like
            The predicted labels or values.
        """
        for component in self._components[:-1]:
            X = component.transform(X)
        return self._components[-1].predict(X)

    def predict_proba(self, X):
        """Make probability predictions using the fitted pipeline.

        Parameters
        ----------
        X : array-like
            The input features for probability prediction.

        Returns
        -------
        array-like
            The predicted probabilities.
        """
        for component in self._components[:-1]:
            X = component.transform(X)
        return self._components[-1].predict_proba(X)
