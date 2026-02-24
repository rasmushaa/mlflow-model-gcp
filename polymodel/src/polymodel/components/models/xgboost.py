import numpy as np
import xgboost as xgb

from ..base_components import BaseModel, TransformMode


class XGBoostModel(BaseModel):
    """Model component that wraps an XGBoost classifier.

    This model can be used for classification tasks and supports the same parameters as XGBoost's XGBClassifier.
    Handles encoding/decoding of string class labels, since XGBoost only supports numeric classes.

    Parameters
    ----------
    features : list[str]
        List of features to use for training the model. Can include wildcards (e.g., "feature_*").
    transform_mode : str or TransformMode
        Mode of transformation: 'replace', 'append', or 'inplace'.
    transform_suffix : str
        Suffix to add to the transformed feature names.
    **kwargs
        Additional keyword arguments to pass to the XGBClassifier constructor.
    """

    def __init__(
        self,
        features: list[str],
        transform_mode: str | TransformMode,
        transform_suffix: str,
        **kwargs,
    ):
        super().__init__(features, transform_mode, transform_suffix)
        self.__model = xgb.XGBClassifier(**kwargs)
        self.__original_classes = None  # Store original class labels (may be strings)
        self.__class_mapping = None  # Map from original to encoded classes
        self.__inverse_mapping = None  # Map from encoded back to original classes

    def _fit_selected(self, X_selected, y):
        """Fit the XGBoost classifier using the selected features and target variable.

        Handles encoding of string class labels to numeric values for XGBoost compatibility.
        """
        # Store original classes and check if encoding is needed
        unique_classes = np.unique(y)
        self.__original_classes = unique_classes.tolist()

        # Check if classes are strings
        if isinstance(unique_classes[0], str):
            # Create mapping from original classes to integers
            self.__class_mapping = {
                cls: i for i, cls in enumerate(sorted(unique_classes))
            }
            self.__inverse_mapping = {i: cls for cls, i in self.__class_mapping.items()}
            # Encode y to integers
            y_encoded = np.array([self.__class_mapping[cls] for cls in y])
            self.__original_classes = sorted(unique_classes)
        else:
            # Classes are already numeric, no encoding needed
            self.__class_mapping = None
            self.__inverse_mapping = None
            y_encoded = y

        self.__model.fit(X_selected, y_encoded)

    def _predict_selected(self, X_selected):
        """Predict class labels for the selected features using the fitted XGBoost classifier.

        Decodes numeric predictions back to original class labels if encoding was used.
        """
        predictions = self.__model.predict(X_selected)

        # Decode predictions if string encoding was used
        if self.__inverse_mapping is not None:
            predictions = np.array(
                [self.__inverse_mapping[pred] for pred in predictions]
            )

        return predictions

    def _predict_proba_selected(self, X_selected):
        """Predict class probabilities for the selected features using the fitted XGBoost classifier."""
        return self.__model.predict_proba(X_selected)

    @property
    def classes(self) -> list:
        """Get the classes recognized by the model.

        Returns
        -------
        list
            A list of class labels (in original format - strings or ints).
        """
        if self.__original_classes is not None:
            return self.__original_classes
        elif hasattr(self.__model, "classes_"):
            return self.__model.classes_.tolist()
        else:
            raise AttributeError(
                "The XGBoostModel has not been fitted yet, so classes are unavailable."
            )
