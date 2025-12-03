


class FeatureSelector:
    """ A basic feature selector that selects specified features from the input data.
    
    Useful for as first component to select a subset of features from a training dataset,
    in scenarios where only certain features are relevant for modeling.
    Possible to use also in production pipelines to ensure only desired features are passed to the model.
    """
    def __init__(self, features):
        """ Initialize the FeatureSelector.

        Parameters
        ----------
        features : list or array-like
            The list of mandatory features to select from the input data.
        """
        self.__features = features

    def __repr__(self):
        return f"{self.__class__.__name__}(features={self.__features!r})"
    
    @property
    def features(self):
        """ Get the list of selected features. 
        
        Returns
        -------
        list
            The list of required features determined in fitting.
        """
        return self.__features

    def fit(self, X, y=None):
        """ A mock fit method for compatibility. Does nothing. """
        pass

    def transform(self, X):
        """ Transform the input data by selecting the specified features. """
        return X[self.__features]
    
    def fit_transform(self, X, y=None):
        """ Fit the selector to the data and then transform it. """
        self.fit(X, y)
        return self.transform(X)