"""
Evaluation Metrics
Last Updated: June 23, 2024
This script supplies the basic functions for evaluating relationships
between data sets. Especially useful for regression analysis.
@author: markjcampmier
"""
# Import Packages
import numpy as np
import pandas as pd
from sklearn.metrics import (r2_score,
                             mean_squared_error,
                             mean_absolute_error,
                             balanced_accuracy_score)


# Define Functions
def adj_r2(y_true, y_hat, k):
    """
    This function calculates the adjusted R^2 score.

    :param y_true: The true values of the target variable.
    :type y_true: numpy.array
    :param y_hat: The predicted values of the target variable.
    :type y_hat: numpy.array
    :param k: The number of features in the model.
    :type k: int
    :returns The adjusted R^2 score.
    :rtype: float
    """
    r2 = r2_score(y_true, y_hat)
    n = len(y_true)
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))


def mae(y_true, y_hat):
    """
    This function calculates the mean absolute error.

    :param y_true: The true values of the target variable.
    :type y_true: numpy.array
    :param y_hat: The predicted values of the target variable.
    :type y_hat: numpy.array
    :return: The mean absolute error.
    :rtype: float
    """
    return mean_absolute_error(y_true, y_hat)


def nmae(y_true, y_hat):
    """
    This function calculates the normalized mean absolute error.

    :param y_true: The true values of the target variable.
    :type y_true: numpy.array
    :param y_hat: The predicted values of the target variable.
    :type y_hat: numpy.array
    :return: The normalized mean absolute error.
    :rtype: float
    """
    return (mae(y_true, y_hat) / np.mean(y_true)) * 100


def rmse(y_true, y_hat):
    """
    This function calculates the root-mean-square error.

    :param y_true: The true values of the target variable.
    :type y_true: numpy.array
    :param y_hat: The predicted values of the target variable.
    :type y_hat: numpy.array
    :return: The root-mean-square error.
    :rtype: float
    """
    mse = mean_squared_error(y_true, y_hat)
    return mse ** 0.5


def nrmse(y_true, y_hat):
    """
    This function calculates the normalized root-mean-square error.

    :param y_true: The true values of the target variable.
    :type y_true: numpy.array
    :param y_hat: The predicted values of the target variable.
    :type y_hat: numpy.array
    :return: The normalized root-mean-square error.
    :rtype: float
    """
    return (rmse(y_true, y_hat) / np.mean(y_true)) * 100


def mbe(y_true, y_hat):
    """
    This function calculates the mean bias error.

    :param y_true: The true values of the target variable.
    :type y_true: numpy.array
    :param y_hat: The predicted values of the target variable.
    :type y_hat: numpy.array
    :return: The mean bias errorr.
    :rtype: float
    """
    return np.mean(y_true - y_hat)


def nmbe(y_true, y_hat):
    """
    This function calculates the normalized mean bias error.

    :param y_true: The true values of the target variable.
    :type y_true: numpy.array
    :param y_hat: The predicted values of the target variable.
    :type y_hat: numpy.array
    :return: The normalized mean bias errorr.
    :rtype: float
    """
    return (mbe(y_true, y_hat) / np.mean(y_true)) * 100


def bas(y_true, y_hat):
    """
    This function calculates the balanced accuracy score.

    :param y_true: The true values of the target variable.
    :type y_true: numpy.array
    :param y_hat: The predicted values of the target variable.
    :type y_hat: numpy.array
    :return: The balanced accuracy score.
    :rtype: float
    """
    return balanced_accuracy_score(y_true, y_hat)


def count_sensors(group):
    """
    Counts the number of sensors in the group.

    :param group: A grouping variable to count the sensors in.
    :type group: pandas.Series
    :return sensors_reporting: Sensors within a given group.
    :rtype sensors_reporting: int
    """
    sensors_reporting = 0
    for sensor in group.unique():
        group_sensor = group.loc[group == sensor]
        if len(group_sensor) >= 168:
            sensors_reporting += 1
    return sensors_reporting


def calculate_r2(group):
    """
    Calculates the R^2 score for a stratified group.

    :param group: A grouping variable to count the sensors in.
    :type group: pandas.Series
    :return: The R^2 score for the group.
    :rtype: float
    """
    if len(group) < 168:
        return -1
    else:
        return r2_score(group['a'], group['b'])


def bias_matrix(df):
    """
    Calculates the matrix-wise normalized mean bias error.

    :param df: A grouping variable to count the sensors in.
    :type df: pandas.DataFrame
    :return matrix: The normalized mean bias for a matrix of groups.
    :rtype matrix: pandas.DataFrame
    """
    columns = df.columns
    matrix = pd.DataFrame(index=columns, columns=columns)

    for i in columns:
        for j in columns:
            bias = nmbe(df.loc[:, i], df.loc[:, j])
            matrix.loc[i, j] = bias

    return matrix
