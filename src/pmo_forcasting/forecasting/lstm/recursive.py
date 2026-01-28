"""

Recursive multi-step forecasting utilities for LSTM models.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def recursive_lstm_forecast(
    model,
    seed_window: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """
    Perform recursive multi-step forecasting with an LSTM model.

    Parameters
    ----------
    model : tf.keras.Model
        Trained LSTM model.
    seed_window : np.ndarray
        Last observed window with shape (1, window_size, 1).
    n_steps : int
        Number of future time steps to forecast.

    Returns
    -------
    np.ndarray
        Forecasted values in scaled space (1D array).
    """
    preds = []
    window = seed_window.copy()

    for _ in range(n_steps):
        next_val = model.predict(window, verbose=0)[0, 0]
        preds.append(next_val)

        # Shift window left and append prediction
        window = np.roll(window, -1, axis=1)
        window[0, -1, 0] = next_val

    return np.array(preds)


def build_forecast_index(
    last_date: pd.Timestamp,
    n_steps: int,
) -> pd.DatetimeIndex:
    """
    Create a business-day forecast index.

    Parameters
    ----------
    last_date : pd.Timestamp
        Last observed date in the test set.
    n_steps : int
        Number of forecast steps.

    Returns
    -------
    pd.DatetimeIndex
        Future business-day index.
    """
    return pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=n_steps
    )


def inverse_scale_forecast(
    forecast_scaled: np.ndarray,
    scaler,
) -> np.ndarray:
    """
    Inverse-transform scaled forecasts to original price space.

    Parameters
    ----------
    forecast_scaled : np.ndarray
        Forecast values in scaled space.
    scaler : MinMaxScaler
        Fitted scaler.

    Returns
    -------
    np.ndarray
        Forecast values in original scale.
    """
    return scaler.inverse_transform(
        forecast_scaled.reshape(-1, 1)
    ).flatten()


def compute_confidence_intervals(
    forecast: pd.Series,
    residuals: np.ndarray,
    z: float = 1.96,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute confidence intervals using residual standard deviation.

    Parameters
    ----------
    forecast : pd.Series
        Point forecast.
    residuals : np.ndarray
        Model residuals from test data.
    z : float
        Z-score (default 1.96 for ~95% CI).

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Upper and lower confidence bounds.
    """
    residual_std = np.std(residuals)
    upper = forecast + z * residual_std
    lower = forecast - z * residual_std
    return lower, upper
