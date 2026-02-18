import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from ..base_components import BaseTransformer, TransformMode


class TextVevtorizer(BaseTransformer):
    """Transformer to vectorize text data in a specified column using CountVectorizer.

    This transformer applies CountVectorizer to the specified text column, creating new features for each token.

    Parameters
    ----------
    features : list[str]
        List of features to transform. Should contain exactly one column name for the text data.
    transform_mode : str or TransformMode
        Mode of transformation: 'replace', 'append', or 'overwrite'.
    transform_suffix : str
        Suffix to add to the transformed feature names.
    kbest : int
        Number of top features to select using chi-squared test.
    """

    def __init__(
        self,
        features: list[str],
        transform_mode: str | TransformMode,
        transform_suffix: str,
        kbest: int,
    ):
        super().__init__(features, transform_mode, transform_suffix)
        self.__vectorizer = CountVectorizer()
        self.__kbest = kbest
        self.__selector = SelectKBest(score_func=chi2, k=self.__kbest)

    def _fit_selected(self, X_selected, y=None):
        """Vectorize the selected text column and select K best features using chi-squared test."""
        if X_selected.shape[1] != 1:
            raise ValueError(
                "TextCountVector expects exactly one feature column to transform.\nGot: "
                + str(X_selected.columns.tolist())
            )

        # Handle missing values and ensure string type
        X_selected = X_selected.fillna("").astype(str)

        # Vectorize categorical features
        X_vectorized = self.__vectorizer.fit_transform(
            X_selected.iloc[:, 0].astype(str)
        ).toarray()

        # Select K best features
        self.__selector.fit(X_vectorized, y)

        # Feature names after vectorization
        self.__feature_names = self.__vectorizer.get_feature_names_out()
        self.__selected_features = self.__feature_names[self.__selector.get_support()]

    def _transform_selected(self, X_selected):
        """Create new features for each token in the selected text column using CountVectorizer and select K best features."""

        # Handle missing values and ensure string type
        X_selected = X_selected.fillna("").astype(str)

        # Vectorize categorical features
        text_vectorized = self.__vectorizer.transform(
            X_selected.iloc[:, 0].astype(str)
        ).toarray()

        # Select K best features
        text_selected = self.__selector.transform(text_vectorized)

        # Selected features as DataFrame
        selected_df = (
            pd.DataFrame(text_selected, columns=self.__selected_features)
            .reset_index(drop=True)
            .astype(float)
        )  # Python int can not contain NaNs, resulting potential unexpected castings aftwards

        return selected_df
