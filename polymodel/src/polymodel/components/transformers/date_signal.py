import numpy as np
import pandas as pd

from ..base_components import BaseTransformer, TransformMode


class DateSignal(BaseTransformer):
    """Transformer to extract date-related signals from a specified datetime column.

    This transformer extracts the following signals from the specified datetime column:
    - Year Sin and Cos (to capture cyclical nature of years)
    - Month Sin and Cos (to capture cyclical nature of months)
    - Day of week (0=Monday, 6=Sunday)
    - Is weekend (boolean)

    Parameters
    ----------
    features : list[str]
        List of features to transform. Should contain exactly one column name for the datetime data.
    transform_mode : str or TransformMode
        Mode of transformation: 'replace', 'append', or 'inplace'.
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
                "DateSignal expects exactly one feature column to transform.\nGot: "
                + str(X_selected.columns.tolist())
            )
        # No fitting needed for this transformer, but we could add checks here if desired.

    def _transform_selected(self, X_selected):
        # Work on a copy of the series; handle missing values
        s = X_selected.iloc[:, 0].ffill().fillna(pd.Timestamp("1970-01-01"))
        s = pd.to_datetime(s, errors="coerce").fillna(pd.Timestamp("1970-01-01"))

        # Extract signals
        month = s.dt.month
        day = s.dt.day
        dayofweek = s.dt.dayofweek
        is_weekend = dayofweek >= 5

        # Create cyclical features for year and month
        year_sin = np.sin(2 * np.pi * month / 12)
        year_cos = np.cos(2 * np.pi * month / 12)
        month_sin = np.sin(2 * np.pi * day / 31)
        month_cos = np.cos(2 * np.pi * day / 31)

        return pd.DataFrame(
            {
                f"{X_selected.columns[0]}_year_sin": year_sin,
                f"{X_selected.columns[0]}_year_cos": year_cos,
                f"{X_selected.columns[0]}_month_sin": month_sin,
                f"{X_selected.columns[0]}_month_cos": month_cos,
                f"{X_selected.columns[0]}_dayofweek": dayofweek,
                f"{X_selected.columns[0]}_is_weekend": is_weekend.astype(int),
            }
        )
