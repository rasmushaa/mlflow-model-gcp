"""
A collection of data processing utilities for ML tasks.
Includes functions to split data into training and testing sets.
"""

from typing import Iterator

import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def split_data(
    data: pd.DataFrame, target_column: str, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into training and testing sets.

    Parameters
    ----------
    data: pd.DataFrame
        The input DataFrame containing features and target
    target_column: str
        The name of the target column in the DataFrame
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        Random seed for reproducibility

    Returns
    -------
    X_train: pd.DataFrame
        Training features
    X_test: pd.DataFrame
        Testing features
    y_train: pd.Series
        Training target
    y_test: pd.Series
        Testing target
    """
    assert (
        target_column in data.columns
    ), f"split_data() Target column '{target_column}' not found in training data: {data.columns.tolist()}"
    assert 0.0 < test_size < 1.0, "split_data() test_size must be between 0.0 and 1.0"
    assert data.shape[0] > 1, "split_data() Data must contain more than one row."

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def kfold_iterator(
    data: pd.DataFrame,
    target_column: str,
    n_splits: int,
    shuffle: bool,
    random_state: int,
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
    random_state: int
        Random seed for reproducibility

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
    assert (
        target_column in data.columns
    ), f"split_data() Target column '{target_column}' not found in training data: {data.columns.tolist()}"
    assert n_splits > 1, "kfold_iterator() n_splits must be greater than 1"
    assert (
        data.shape[0] > n_splits
    ), "kfold_iterator() Data must contain more rows than n_splits."

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        yield fold, X_train, X_val, y_train, y_val
