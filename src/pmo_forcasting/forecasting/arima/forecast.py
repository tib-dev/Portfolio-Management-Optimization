# forecasting/arima/forecast.py
"""
ARIMA forecasting utilities.

Provides functions to generate out-of-sample predictions
from trained ARIMA/SARIMA models.
"""

from typing import Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def forecast_arima(
    model: Any,
    n_periods: int,
    index: Optional[pd.DatetimeIndex] = None,
    return_conf_int: bool = False,
) -> pd.Series | Tuple[pd.Series, pd.DataFrame]:
    """
    Generate forecasts using a trained ARIMA/SARIMA model.

    Parameters
    ----------
    model : pmdarima.arima.ARIMA
        Fitted ARIMA/SARIMA model.
    n_periods : int
        Number of periods to forecast.
    index : pd.DatetimeIndex, optional
        Optional datetime index for the forecasted values.
    return_conf_int : bool, default=False
        Whether to return confidence intervals.

    Returns
    -------
    pd.Series or (pd.Series, pd.DataFrame)
        Forecasted values, optionally with confidence intervals.
    """
    try:
        if return_conf_int:
            preds, conf_int = model.predict(
                n_periods=n_periods, return_conf_int=True
            )
        else:
            preds = model.predict(n_periods=n_periods)
            conf_int = None

        preds = np.asarray(preds, dtype=float)

        # ---- Index handling ----
        if index is not None:
            if len(index) >= n_periods:
                index = index[:n_periods]
            else:
                logger.warning(
                    "Index shorter than n_periods. Falling back to RangeIndex."
                )
                index = pd.RangeIndex(n_periods)
        else:
            index = pd.RangeIndex(n_periods)

        preds_series = pd.Series(preds, index=index, name="forecast")

        if return_conf_int:
            conf_df = pd.DataFrame(
                conf_int,
                index=index,
                columns=["lower", "upper"],
            )
            return preds_series, conf_df

        logger.info("ARIMA forecast generated for %d periods", n_periods)
        return preds_series

    except Exception as e:
        logger.exception("ARIMA forecasting failed")
        raise RuntimeError(f"ARIMA forecasting failed: {e}") from e
