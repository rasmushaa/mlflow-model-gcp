from sklearn.ensemble import RandomForestClassifier


class RandomForestModel(RandomForestClassifier):
    """Random Forest Classifier model.

    Inherits from sklearn.ensemble.RandomForestClassifier.
    """

    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)

    def __repr__(self) -> str:
        return str(super())

    @property
    def features(self) -> list[str]:
        """Get the feature names used by the model.

        Returns
        -------
        list
            A list of feature names.
        """
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_.tolist()
        else:
            raise AttributeError(
                "The model has not been fitted yet, so feature names are unavailable."
            )

    @property
    def classes(self) -> list:
        """Get the classes recognized by the model.

        Returns
        -------
        list
            A list of class labels.
        """
        if hasattr(self, "classes_"):
            return self.classes_.tolist()
        else:
            raise AttributeError(
                "The model has not been fitted yet, so classes are unavailable."
            )
