# forecasting/lstm/forecast.py
"""
LSTM forecasting utilities.

Provides functions to generate predictions from a trained LSTM model,
including inverse scaling support.
"""

from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)


def forecast_lstm(model: Any,
                  X_test: np.ndarray,
                  scaler: Optional[MinMaxScaler] = None) -> np.ndarray:
    """
    Generate forecasts using a trained LSTM model.

    Parameters
    ----------
    model : tf.keras.Model
        Trained LSTM model.
    X_test : np.ndarray
        Input sequences for forecasting.
    scaler : MinMaxScaler, optional
        Scaler used during training. If provided, the predictions
        are inverse-transformed to the original scale.

    Returns
    -------
    np.ndarray
        Predicted values, scaled back if scaler is provided.
    """
    try:
        preds = model.predict(X_test).flatten()
        if scaler:
            preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        logger.info("LSTM forecast generated for %d steps", len(preds))
        return preds
    except Exception as e:
        logger.exception("LSTM forecasting failed")
        raise RuntimeError(f"LSTM forecasting failed: {e}")
