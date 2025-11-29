"""
A collection of data processing utilities for ML tasks.
Includes functions to split data into training and testing sets.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    data: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    ''' Split the data into training and testing sets.

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
    '''
    assert target_column in data.columns, f"split_data() Target column '{target_column}' not found in training data: {data.columns.tolist()}"
    assert 0.0 < test_size < 1.0, "split_data() test_size must be between 0.0 and 1.0"
    assert data.shape[0] > 1, "split_data() Data must contain more than one row."

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test