import pandas as pd


class FeatureSelector:
    """A basic feature selector that selects specified features from the input data.

    Useful for as first component to select a subset of features from a training dataset,
    in scenarios where only certain features are relevant for modeling.
    Possible to use also in production pipelines to ensure only desired features are passed to the model.
    """

    def __init__(self, features):
        """Initialize the FeatureSelector.

        Parameters
        ----------
        features : list or array-like
            The list of mandatory features to select from the input data.
        """
        self.__signature = []
        self.__features = features

    def __repr__(self):
        return f"{self.__class__.__name__}(features={self.__features!r})"

    @property
    def signature(self) -> list[str]:
        """Get the list of input features signature."""
        return self.__signature

    @property
    def features(self) -> list[str]:
        """Get the list of used features."""
        return self.__features

    def fit(self, X: pd.DataFrame, y=None) -> None:
        self.__signature = X.columns.tolist()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data by selecting the specified features."""
        return X[self.__features]

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit the selector to the data and then transform it."""
        self.fit(X, y)
        return self.transform(X)
