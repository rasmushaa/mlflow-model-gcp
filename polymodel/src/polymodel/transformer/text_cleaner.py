class TextCleaner:
    """A transformer that cleans text data in a specified column."""

    def __init__(self, text_column):
        """Initialize the TextCleaner.

        Parameters
        ----------
        text_column : str
            The name of the text column to clean.
        """
        self.__signature = []
        self.__text_column = text_column

    def __repr__(self):
        return f"{self.__class__.__name__}(text_column={self.__text_column!r})"

    @property
    def signature(self) -> list[str]:
        """Get the list of input features signature."""
        return self.__signature

    @property
    def features(self) -> list[str]:
        """Get the list of features used by the transformer.

        Returns
        -------
        list
            The list of features determined in fitting.
        """
        return [self.__text_column]

    def fit(self, X, y=None):
        """Fit the transformer to the input DataFrame."""
        self.__signature = X.columns.tolist()

    def transform(self, X):
        """Transform the input data by cleaning the text column."""
        X_cleaned = X.copy()

        if self.__text_column not in X_cleaned.columns:
            raise KeyError(
                f"Text column {self.__text_column!r} not found in input DataFrame"
            )

        # Work on a copy of the series; handle missing values
        s = X_cleaned[self.__text_column].fillna("").astype(str)

        # Lowercase, remove any non-alphabetic characters (including numbers),
        # collapse multiple spaces and strip ends.
        s = (
            s.str.lower()
            .str.replace(r"[^a-z\s]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # Remove one-character tokens (e.g., single letters) while preserving
        s = s.str.split().apply(lambda toks: " ".join([t for t in toks if len(t) > 1]))

        X_cleaned[self.__text_column] = s
        return X_cleaned

    def fit_transform(self, X, y=None):
        """Fit the transformer to the data and then transform it."""
        self.fit(X, y)
        return self.transform(X)
