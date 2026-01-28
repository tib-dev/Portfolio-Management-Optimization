import numpy as np
import pandas as pd
from typing import Dict


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    Safe for financial series by masking zero values.
    """
    mask = y_true != 0
    if not np.any(mask):
        return np.nan

    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


def evaluate(y_true, y_pred) -> Dict[str, float]:
    """
    Evaluate forecasting predictions with NaN-safe and
    time-series-safe metrics.

    Automatically aligns pandas Series by index.
    """
    # ---- Align if pandas objects ----
    if isinstance(y_true, (pd.Series, pd.DataFrame)) or isinstance(
        y_pred, (pd.Series, pd.DataFrame)
    ):
        y_true = pd.Series(y_true)
        y_pred = pd.Series(y_pred)

        y_true, y_pred = y_true.align(y_pred, join="inner")

        y_true = y_true.values
        y_pred = y_pred.values
    else:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

    # ---- Remove NaNs safely ----
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}

    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
