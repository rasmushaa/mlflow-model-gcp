import fnmatch
import logging
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TransformMode(str, Enum):
    """Methods for combining transformed features with the original dataset.

    - `inplace`: Replace the original features with the transformed ones.
    - `append`: Add the transformed features to the original dataset without removing any columns.
    - `replace`: Use only the transformed features, discarding all original features.
    """

    INPLACE = "inplace"
    APPEND = "append"
    REPLACE = "replace"


class BaseComponent(ABC):
    """Common behavior for all pipeline components."""

    def __init__(
        self,
        features: list[str],
        transform_mode: str | TransformMode,
        transform_suffix: str,
    ) -> None:
        """Init the base component with common parameters.

        Parameters
        ----------
        features : list[str]
            A list of feature patterns to select from the input DataFrame.
            Patterns can include wildcards (e.g., `feature_*`) to match multiple columns.
            If `None`, all features will be selected. Default is `None`.
        transform_mode : str | TransformMode
            The method for combining transformed features with the original dataset.
        transform_suffix : str
            A suffix to append to the names of transformed features when using `append` mode.
        """
        self._features = features or ["*"]
        self._transform_mode = TransformMode(transform_mode)
        self._transform_suffix = transform_suffix
        self._signature: list[str] = []
        self._resolved_features: list[str] = []

    @property
    def signature(self) -> list[str]:
        """Get the list of input features signature.

        Returns
        -------
        list[str]
            A list of feature names that have been seen during fitting.
        """
        if not self._signature:
            raise ValueError(
                "Component has not been fitted yet, signature is not available."
            )
        return self._signature

    @property
    def resolved_features(self) -> list[str]:
        """Get the list of resolved features that this component actually uses.

        Returns
        -------
        list[str]
            A list of feature names that have been resolved from the specified patterns.
        """
        if not self._resolved_features:
            raise ValueError(
                "Component has not been fitted yet, resolved features are not available."
            )
        return self._resolved_features

    def _get_selected_features(self, X: pd.DataFrame) -> list[str]:
        """Find the columns in `X` that match the patterns specified in `self._features`.

        This method resolves the feature patterns (which may include wildcards) against the columns of `X`.
        First it checks if the pattern contains any wildcard characters.
        If it does, it uses `fnmatch` to find all matching columns,
        otherwise it checks for an exact match.
        Each column is included only once in the final list of resolved features.

        Raises
        ------
        KeyError
            If any feature pattern does not match any columns in `X`, a KeyError is raised

        Returns
        -------
        list[str]
            A unique list of column names from `X` that match the specified feature patterns.
        """
        columns = X.columns.tolist()
        resolved: list[str] = []

        for pattern in self._features:
            is_wildcard = any(char in pattern for char in ("*", "?", "["))
            if is_wildcard:
                matches = [
                    column for column in columns if fnmatch.fnmatchcase(column, pattern)
                ]
            else:
                matches = [pattern] if pattern in columns else []

            if not matches:
                raise KeyError(
                    f"Feature pattern `{pattern}` did not match any columns: {list(X.columns)}"
                )

            for match in matches:
                if match not in resolved:
                    resolved.append(match)

        return resolved

    def _apply_transform(
        self,
        X: pd.DataFrame,
        transformed: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply the specified `transform_mode` to combine the transformed features with the original dataset `X`.

        Parameters
        ----------
        X : pd.DataFrame
            The original input dataset.
        transformed : pd.DataFrame
            The DataFrame containing the transformed features.
        """

        # Rename transformed columns by appending the specified suffix.
        transformed.columns = [
            f"{col}{self._transform_suffix}" for col in transformed.columns
        ]

        # If `replace`, return only the transformed features.
        if self._transform_mode == TransformMode.REPLACE:
            return transformed.copy()

        # If `inplace`, drop the original features from `X` and keep the rest.
        if self._transform_mode == TransformMode.INPLACE:
            base = X.drop(columns=self._resolved_features).reset_index(drop=True)

        # If `append`, keep all original features and add the transformed ones.
        else:
            base = X.copy().reset_index(drop=True)

        # Check for duplicate columns between the base and transformed DataFrames.
        duplicate_columns = [
            column for column in transformed.columns if column in base.columns
        ]
        if duplicate_columns:
            joined = ", ".join(duplicate_columns)
            raise ValueError(
                f"Transformed features produced duplicate columns: {joined}. "
                "Use unique output names or `overwrite` mode when replacing originals."
            )

        return pd.concat([base, transformed], axis=1)


class BaseTransformer(BaseComponent):
    """Base transformer: receives full dataset, transforms selected features.

    Methods
    -------
    fit
        Fit the transformer to the input dataset `X` and target `y`, using only the selected features.
    transform
        Transform the input dataset `X` using only the selected features, and combine with the original dataset according to the specified `transform_mode`.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BaseTransformer":
        """Public method to fit the transformer to the input dataset `X` and target `y`.

        Only the selected features are passed to the internal `_fit_selected` method,
        which is responsible for fitting the transformer using only the selected features.

        Parameters
        ----------
        X : pd.DataFrame
            The full input dataset from which to select features for fitting.
        y : pd.Series | None, optional
            The target variable for supervised fitting. Default is `None` for unsupervised transformers.

        Returns
        -------
        BaseTransformer
            The fitted transformer instance (self).
        """
        self._signature = X.columns.tolist()
        self._resolved_features = self._get_selected_features(X)
        selected = X[self._resolved_features]
        logger.debug(
            f"Fitting {self.__class__.__name__} with: {selected.shape}\n{selected.head()}"
        )
        self._fit_selected(selected, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Public method to transform the input dataset `X`.

        Only the selected features are passed to the internal `_transform_selected` method,
        which is responsible for generating the transformed features.

        Parameters
        ----------
        X : pd.DataFrame
            The full input dataset from which to select features for transformation.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the transformed features,
            according to the specified `transform_mode` for combining with the original dataset.
        """
        transformed = self._transform_selected(X[self._resolved_features].copy())
        transformed.reset_index(drop=True, inplace=True)
        return self._apply_transform(X, transformed)

    @abstractmethod
    def _fit_selected(
        self, X_selected: pd.DataFrame, y: pd.Series | None = None
    ) -> None:
        raise NotImplementedError(
            "Subclasses must implement the `_fit_selected` method to fit the transformer using only the selected features from the input dataset."
        )

    @abstractmethod
    def _transform_selected(self, X_selected: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "Subclasses must implement the `_transform_selected` method to return a DataFrame of transformed features based on the selected input features."
        )


class BaseModel(BaseTransformer):
    """Model is a specialized transformer wtith:

    Methods
    -------
    predict
        Predict class labels based on the selected features.
    predict_proba
        Predict class probabilities based on the selected features.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Public method to predict class labels.

        Only the selected features are passed to the internal `_predict_selected` method,
        which is responsible for generating the predicted class labels.

        Parameters
        ----------
        X : pd.DataFrame
            The full input dataset from which to select features for prediction.

        Returns
        -------
        np.ndarray
            An array of predicted class labels.
        """
        selected = self._get_selected_features(X)
        predictions = self._predict_selected(X[selected])
        if isinstance(predictions, pd.Series):
            return predictions.to_numpy()
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Public method to predict probabilities

        Only the selected features are passed to the internal `_predict_proba_selected` method,
        which is responsible for generating the predicted probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            The full input dataset from which to select features for probability prediction.

        Returns
        -------
        np.ndarray
            An array of predicted probabilities for each class.
        """
        selected = self._get_selected_features(X)
        probabilities = self._predict_proba_selected(X[selected])
        if isinstance(probabilities, pd.DataFrame):
            return probabilities.to_numpy()
        return probabilities

    def _transform_selected(self, X_selected: pd.DataFrame) -> pd.DataFrame:
        """The default model transformation is to output the predicted probabilities as new features.

        The new columns are named using the internal `classes`,
        and appending the specified suffix.

        Parameters
        ----------
        X_selected : pd.DataFrame
            The subset of the original DataFrame containing only the selected features for transformation.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the predicted probabilities for each class,
            with appropriately suffixed column names.
        """
        preds = self.predict_proba(X_selected)
        return pd.DataFrame(preds, columns=self.classes)

    @property
    @abstractmethod
    def classes(self) -> list[str]:
        raise NotImplementedError(
            "Subclasses must implement the `classes` property to return the list of class labels."
        )

    @abstractmethod
    def _predict_selected(self, X_selected: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            "Subclasses must implement the `_predict_selected` method to return predictions based on the selected features."
        )

    @abstractmethod
    def _predict_proba_selected(self, X_selected: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            "Subclasses must implement the `_predict_proba_selected` method to return predicted probabilities based on the selected features."
        )
