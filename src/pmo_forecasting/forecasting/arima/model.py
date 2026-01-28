# forecasting/arima/model.py
"""
ARIMA model builder and trainer for time series forecasting.

This module provides:
- build_arima: Constructs an ARIMA/SARIMA model based on configuration.
- train_arima: Fits the ARIMA model on training data.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from pmdarima import auto_arima
import logging

logger = logging.getLogger(__name__)


def build_arima(y_train: pd.Series, cfg: Dict[str, Any]):
    """
    Build an ARIMA or SARIMA model using pmdarima.auto_arima.

    Parameters
    ----------
    y_train : pd.Series
        Historical time series data for training.
    cfg : dict
        ARIMA configuration dictionary. Can include keys:
        - p, d, q: ARIMA orders
        - P, D, Q, m: Seasonal orders
        - seasonal : bool, whether to fit SARIMA
        - max_p, max_q, max_d, etc. for auto_arima search

    Returns
    -------
    pmdarima.arima.ARIMA
        Untrained ARIMA model specification ready for fitting.
    """
    try:
        seasonal = cfg.get("seasonal", False)
        m = cfg.get("m", 0) if seasonal else 0

        # Auto-determine d if not provided
        d = cfg.get("d", None)

        model = auto_arima(
            y=y_train,
            start_p=cfg.get("p", 1),
            start_q=cfg.get("q", 1),
            max_p=cfg.get("max_p", 5),
            max_q=cfg.get("max_q", 5),
            d=d,
            seasonal=seasonal,
            start_P=cfg.get("P", 0),
            start_Q=cfg.get("Q", 0),
            max_P=cfg.get("max_P", 2),
            max_Q=cfg.get("max_Q", 2),
            D=cfg.get("D", 0),
            m=m,
            stepwise=cfg.get("stepwise", True),
            suppress_warnings=True,
            error_action="ignore",
            trace=cfg.get("trace", False),
        )

        logger.info("ARIMA model built: %s", model.summary())
        return model

    except Exception as e:
        logger.exception("Failed to build ARIMA model")
        raise RuntimeError(f"ARIMA building failed: {e}")


def train_arima(model: Any, y_train: pd.Series):
    """
    Fit the ARIMA model on training data.

    Parameters
    ----------
    model : pmdarima.arima.ARIMA
        ARIMA model returned from build_arima.
    y_train : pd.Series
        Historical training series.

    Returns
    -------
    pmdarima.arima.ARIMA
        Fitted ARIMA model.
    """
    try:
        model.fit(y_train)
        logger.info("ARIMA model trained successfully")
        return model
    except Exception as e:
        logger.exception("ARIMA training failed")
        raise RuntimeError(f"ARIMA training failed: {e}")
