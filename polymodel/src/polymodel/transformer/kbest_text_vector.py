import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2


class KbestTextVector:
    """A preprocessor that vectorizes a text column using CountVectorizer,
    and selects the K best features using chi-squared statistical test.
    """

    def __init__(self, text_column: str, kbest: int):
        """Initialize the KbestTextVector processor.

        Parameters
        ----------
        text_column: str
            The name of the text column to be replaced by CountVectorizer features.
        kbest: int
            The number of best features to select.
        """
        self.__text_column = text_column
        self.__kbest = kbest
        self.__signature: list[str] = []
        self.__selector = SelectKBest(score_func=chi2, k=self.__kbest)
        self.__vectorizer = CountVectorizer()

    def __repr__(self):
        return f"{self.__class__.__name__}(text_column={self.__text_column!r}, kbest={self.__kbest!r})"

    @property
    def signature(self) -> list[str]:
        """Get the list of input features signature.

        Returns
        -------
        list
            The list of input features signature. Empty before fitting.
        """
        return self.__signature

    @property
    def features(self) -> list[str]:
        """Get the list of selected features after fitting.

        Returns
        -------
        list
            The list of selected features determined in fitting.
        """
        return [self.__text_column]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the processor to the input DataFrame.

        Parameters
        ----------
        X: pd.DataFrame
            Input DataFrame to fit.
        y: pd.Series
            Target variable for feature selection.
        """
        self.__signature = X.columns.tolist()

        X = X[self.__text_column]
        y = y

        # Vectorize categorical features
        X_vectorized = self.__vectorizer.fit_transform(X.astype(str)).toarray()

        # Select K best features
        self.__selector.fit(X_vectorized, y)

        # Feture names after vectorization
        self.__feature_names = self.__vectorizer.get_feature_names_out()
        self.__selected_features = self.__feature_names[self.__selector.get_support()]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input DataFrame using the fitted processor.

        The specified text column is replaced with the selected K best features,
        on the same position as the original column.

        Parameters
        ----------
        X: pd.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with selected features.
        """
        text_data = X[self.__text_column]

        # Vectorize categorical features
        text_vectorized = self.__vectorizer.transform(text_data.astype(str)).toarray()

        # Select K best features
        text_selected = self.__selector.transform(text_vectorized)

        # Insert selected features back into DataFrame at original column position
        orig_idx = X.columns.get_loc(self.__text_column)

        # Split original frame into left (before text_col) and right (after text_col)
        left_df = X.iloc[:, :orig_idx].reset_index(drop=True)
        right_df = X.iloc[:, orig_idx + 1 :].reset_index(drop=True)

        # Selected features as DataFrame
        selected_df = pd.DataFrame(
            text_selected, columns=self.__selected_features
        ).reset_index(drop=True)

        # Concatenate: left + selected features (starting at orig_idx) + original right columns (kept last)
        transformed_df = pd.concat([left_df, selected_df, right_df], axis=1)

        return transformed_df.astype(
            float
        )  # Python int can not contain NaNs, resulting potential unexpected castings aftwards

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit the processor to the input DataFrame and then transform it.

        Parameters
        ----------
        X: pd.DataFrame
            Input DataFrame to fit and transform.
        y: pd.Series
            Target variable for feature selection.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with selected features.
        """
        self.fit(X, y)
        return self.transform(X)
