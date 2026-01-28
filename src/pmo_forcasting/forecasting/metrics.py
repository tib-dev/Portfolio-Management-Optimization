import numpy as np


def mae(y_true, y_pred):
    """
    Compute Mean Absolute Error (MAE).

    MAE measures the average magnitude of errors
    without considering their direction.

    Returns
    -------
    float
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE).

    RMSE penalizes large errors more heavily
    and is sensitive to outliers.

    Returns
    -------
    float
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error (MAPE).

    Expresses forecast error as a percentage,
    useful for relative performance comparison.

    Returns
    -------
    float
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
