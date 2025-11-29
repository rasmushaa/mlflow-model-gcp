


class FeatureSelector:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        return X[self.features]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)