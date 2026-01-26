from .metrics import mae, rmse, mape


def evaluate(y_true, y_pred):
    """
    Evaluate forecast accuracy using multiple metrics.

    Parameters
    ----------
    y_true : array-like
        Actual observed values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
