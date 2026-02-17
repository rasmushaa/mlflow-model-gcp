import pandas as pd

from ..base_components import BaseTransformer, TransformMode


class TextCleaner(BaseTransformer):
    """Transformer to clean text data in a specified column.

    This transformer performs the following cleaning steps on the specified text column:
    - Converts text to lowercase.
    - Removes non-alphabetic characters (including numbers).
    - Collapses multiple spaces into a single space.
    - Strips leading and trailing whitespace.
    - Removes one-character tokens (e.g., single letters) while preserving valid words.

    Parameters
    ----------
    features : list[str]
        List of features to transform. Should contain exactly one column name for the text data.
    transform_mode : str or TransformMode
        Mode of transformation: 'replace', 'append', or 'prepend'.
    transform_suffix : str
        Suffix to add to the transformed feature names
    """

    def __init__(
        self,
        features: list[str],
        transform_mode: str | TransformMode,
        transform_suffix: str,
    ):
        super().__init__(features, transform_mode, transform_suffix)

    def _fit_selected(self, X_selected, y=None):
        if X_selected.shape[1] != 1:
            raise ValueError(
                "TextCleaner expects exactly one feature column to transform.\nGot: "
                + str(X_selected.columns.tolist())
            )
        # No fitting needed for this transformer, but we could add checks here if desired.

    def _transform_selected(self, X_selected):
        # Work on a copy of the series; handle missing values
        s = X_selected.iloc[:, 0].fillna("").astype(str)

        # Lowercase, remove any non-alphabetic characters (including numbers),
        # collapse multiple spaces and strip ends.
        s = (
            s.str.lower()
            .str.replace(r"[^a-z\s]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # Remove one-character tokens (e.g., single letters) while preserving valid words
        s = s.str.split().apply(lambda toks: " ".join([t for t in toks if len(t) > 1]))

        return pd.DataFrame(s, columns=[X_selected.columns[0]])
