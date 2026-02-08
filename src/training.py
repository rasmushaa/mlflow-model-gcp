from typing import Iterator

import pandas as pd
from sklearn.model_selection import KFold


def kfold_iterator(
    data: pd.DataFrame,
    target_column: str,
    features: list[str],
    n_splits: int,
    shuffle: bool,
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """Generate K-Fold cross-validation splits.

    Parameters
    ----------
    data: pd.DataFrame
        The input DataFrame containing features and target
    target_column: str
        The name of the target column in the DataFrame
    n_splits: int
        Number of folds
    shuffle: bool
        Whether to shuffle data before splitting

    Yields
    ------
    fold: int
        Fold index
    X_train: pd.DataFrame
        Training features for the fold
    X_val: pd.DataFrame
        Validation features for the fold
    y_train: pd.Series
        Training target for the fold
    y_val: pd.Series
        Validation target for the fold
    """
    if target_column not in data.columns:
        raise ValueError(
            f"kfold_iterator() Target column '{target_column}' not found in data: {data.columns.tolist()}"
        )
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        raise ValueError(
            f"kfold_iterator() Features not found in data: {missing_features}"
        )
    if n_splits <= 1:
        raise ValueError("kfold_iterator() n_splits must be greater than 1")
    if data.shape[0] <= n_splits:
        raise ValueError("kfold_iterator() Data must contain more rows than n_splits.")

    if shuffle:
        random_state = 42  # Fixed random state for reproducibility when shuffling
    else:
        random_state = None

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    X = data[features]
    y = data[target_column]

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        yield fold, X_train, X_val, y_train, y_val
