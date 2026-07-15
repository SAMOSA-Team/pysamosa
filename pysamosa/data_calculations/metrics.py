"""Evaluation metrics for regression and classification model assessment."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    balanced_accuracy_score,
)


def adj_r2(y_true: np.ndarray, y_hat: np.ndarray, k: int) -> float:
    """Calculate the adjusted R² score.

    Args:
        y_true: True values.
        y_hat: Predicted values.
        k: Number of features in the model.

    Returns:
        Adjusted R² score.
    """
    r2 = r2_score(y_true, y_hat)
    n = len(y_true)
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))


def mae(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculate the mean absolute error.

    Args:
        y_true: True values.
        y_hat: Predicted values.

    Returns:
        Mean absolute error.
    """
    return mean_absolute_error(y_true, y_hat)


def nmae(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculate the normalized mean absolute error [%].

    Args:
        y_true: True values.
        y_hat: Predicted values.

    Returns:
        Normalized mean absolute error.
    """
    return (mae(y_true, y_hat) / np.mean(y_true)) * 100


def rmse(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculate the root-mean-square error.

    Args:
        y_true: True values.
        y_hat: Predicted values.

    Returns:
        Root-mean-square error.
    """
    return mean_squared_error(y_true, y_hat) ** 0.5


def nrmse(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculate the normalized root-mean-square error [%].

    Args:
        y_true: True values.
        y_hat: Predicted values.

    Returns:
        Normalized root-mean-square error.
    """
    return (rmse(y_true, y_hat) / np.mean(y_true)) * 100


def mbe(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculate the mean bias error.

    Args:
        y_true: True values.
        y_hat: Predicted values.

    Returns:
        Mean bias error.
    """
    return np.mean(y_true - y_hat)


def nmbe(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculate the normalized mean bias error [%].

    Args:
        y_true: True values.
        y_hat: Predicted values.

    Returns:
        Normalized mean bias error.
    """
    return (mbe(y_true, y_hat) / np.mean(y_true)) * 100


def bas(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculate the balanced accuracy score.

    Args:
        y_true: True class labels.
        y_hat: Predicted class labels.

    Returns:
        Balanced accuracy score.
    """
    return balanced_accuracy_score(y_true, y_hat)


def count_sensors(group: pd.Series) -> int:
    """Count the number of sensors in a group that report at least 168 observations.

    Args:
        group: Series of sensor identifiers.

    Returns:
        Count of qualifying sensors.
    """
    sensors_reporting = 0
    for sensor in group.unique():
        if len(group.loc[group == sensor]) >= 168:
            sensors_reporting += 1
    return sensors_reporting


def calculate_r2(group: pd.DataFrame) -> float:
    """Calculate R² for a stratified group (requires columns 'a' and 'b').

    Args:
        group: DataFrame with at least 168 rows and columns 'a' and 'b'.

    Returns:
        R² score, or -1 if insufficient data.
    """
    if len(group) < 168:
        return -1
    return r2_score(group["a"], group["b"])


def bias_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate a matrix of normalized mean bias errors between all column pairs.

    Args:
        df: DataFrame whose columns represent sensors or sites.

    Returns:
        Square matrix of NMBE values.
    """
    columns = df.columns
    matrix = pd.DataFrame(index=columns, columns=columns)
    for i in columns:
        for j in columns:
            matrix.loc[i, j] = nmbe(df.loc[:, i], df.loc[:, j])
    return matrix
