from sklearn.ensemble import RandomForestClassifier

from ..base_components import BaseModel, TransformMode


class NaiveBayesModel(BaseModel):
    """Model component that wraps a Naive Bayes classifier from scikit-learn.

    This model can be used for classification tasks and supports the same parameters as sklearn's Naive Bayes classifiers (e.g., MultinomialNB, GaussianNB).

    Parameters
    ----------
    features : list[str]
        List of features to use for training the model. Can include wildcards (e.g., "feature_*").
    transform_mode : str or TransformMode
        Mode of transformation: 'replace', 'append', or 'inplace'.
    transform_suffix : str
        Suffix to add to the transformed feature names.
    **kwargs
        Additional keyword arguments to pass to the Naive Bayes classifier constructor.
    """

    def __init__(
        self,
        features: list[str],
        transform_mode: str | TransformMode,
        transform_suffix: str,
        **kwargs,
    ):
        super().__init__(features, transform_mode, transform_suffix)
        self.__model = RandomForestClassifier(**kwargs)

    def _fit_selected(self, X_selected, y):
        """Fit the Naive Bayes classifier using the selected features and target variable."""
        self.__model.fit(X_selected, y)

    def _predict_selected(self, X_selected):
        """Predict class labels for the selected features using the fitted Naive Bayes classifier."""
        return self.__model.predict(X_selected)

    def _predict_proba_selected(self, X_selected):
        """Predict class probabilities for the selected features using the fitted Naive Bayes classifier."""
        return self.__model.predict_proba(X_selected)

    @property
    def classes(self) -> list:
        """Get the classes recognized by the model.

        Returns
        -------
        list
            A list of class labels.
        """
        if hasattr(self.__model, "classes_"):
            return self.__model.classes_.tolist()
        else:
            raise AttributeError(
                "The NaiveBayesModel has not been fitted yet, so classes are unavailable."
            )
