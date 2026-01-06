"""
A module defining a machine learning pipeline that integrates data transformers and a model.
This pipeline allows for sequential data preprocessing followed by model training and prediction.
A interfaces for transformers and models are defined using Protocols to ensure compatibility.
"""

import logging
from typing import List

import pandas as pd

from .model.interface import ModelInterface
from .transformer.interface import TransformerInterface

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self, transformers: List[TransformerInterface], model: ModelInterface
    ) -> None:
        """Initializes the Pipeline with a list of transformers and a model.

        Parameters
        ----------
        transformers : List[TransformerInterface]
            A sequence of transformer instances to preprocess the data in sequence.
        model : ModelInterface
            The machine learning model to be trained and used for predictions.
        """
        self.__transformers = transformers
        self.__model = model
        self.__signature: List[str] = []

    def __repr__(self) -> str:
        """Return a string representation of the pipeline,
        including its transformers and model.
        """
        str = "\nPipeline("
        str += "\n  Transformers: ["
        for transformer in self.__transformers:
            str += f"\n    {transformer!r},"
        str += "\n  ]"
        str += "\n  Model:"
        str += f"\n    {self.__model!r}"
        str += "\n)"
        return str

    @property
    def model(self) -> ModelInterface:
        """Get the model used in the pipeline.

        Returns
        -------
        ModelInterface
            The machine learning model instance.
        """
        return self.__model

    @property
    def architecture(self) -> str:
        """Get a string representation of the pipeline architecture.

        Returns
        -------
        str
            A string describing the sequence of transformers and the model in the pipeline.
        """
        arch = "->".join([t.__class__.__name__ for t in self.__transformers])
        arch += "->" + self.__model.__class__.__name__
        return arch

    @property
    def layers(self) -> dict:
        """Get the signatures of the transformers and model in the pipeline.

        Returns
        -------
        dict
            A dictionary containing the names and features of each transformer and the model.
        """
        signatures = {}
        for i, transformer in enumerate(self.__transformers):
            entry = {
                "name": transformer.__class__.__name__,
                "signature": transformer.signature,
                "features": transformer.features,
            }
            signatures[i] = entry
        signatures[len(self.__transformers)] = {
            "name": self.__model.__class__.__name__,
            "features": self.__model.features,
        }
        return signatures

    @property
    def signature(self) -> List[str]:
        """Get the input signature of the pipeline.

        The input signature is determined during the fitting process and represents
        the features expected by the pipeline.

        Returns
        -------
        List[str]
            A list of feature names that form the input signature of the pipeline.
        """
        return self.__signature

    @property
    def features(self) -> List[str]:
        """Get the list of actually required features by the pipeline.

        Depending on the architecture of the pipeline, some features might be
        transformed or dropped by the transformers. This property returns the final
        list of mandatory features required by the pipeline.

        Details
        -------
        The features are determined by checking which features from the input
        signature are used in at least one layer of the pipeline.
        Note, some layers may create new features that are not part of the input signature.

        Returns
        -------
        List[str]
            A list of feature names that are required by the pipeline.
        """
        all_features = []
        for layer in self.layers.values():
            all_features.extend(layer.get("features", []))
        return [feature for feature in self.__signature if feature in all_features]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fits the transformers and the model on the training data.

        Parameters
        ----------
        X : pd.DataFrame
            The input features for training.
        y : pd.Series
            The target labels for training.
        """
        self.__signature = X.columns.tolist()
        logger.info("Start pipeline training")
        for transformer in self.__transformers:
            X = transformer.fit_transform(X, y)
        self.__model.fit(X, y)
        logger.info("End pipeline training")
        logger.debug(f"Trained model pipeline layers:\n{self.layers}")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Makes predictions using the trained model after transforming the input data.

        Parameters
        ----------
        X : pd.DataFrame
            The input features for making predictions.

        Returns
        -------
        pd.Series
            The predicted labels.
        """
        for transformer in self.__transformers:
            X = transformer.transform(X)
        return self.__model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Makes probability predictions using the trained model after transforming the input data.

        Parameters
        ----------
        X : pd.DataFrame
            The input features for making probability predictions.

        Returns
        -------
        pd.DataFrame
            The predicted probabilities for each class.
        """
        for transformer in self.__transformers:
            X = transformer.transform(X)
        return self.__model.predict_proba(X)
