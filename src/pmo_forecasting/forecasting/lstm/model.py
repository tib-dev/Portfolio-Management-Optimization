"""
forecasting/lstm/model.py
Improved LSTM model builder supporting dynamic layer configurations.
"""
from __future__ import annotations
import logging
from typing import Tuple, Dict, Any, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

logger = logging.getLogger(__name__)


def build_lstm(input_shape: Tuple[int, int], cfg: Dict) -> Sequential:
    """
    Build a stacked LSTM model based on a dynamic list of hidden units.
    """
    try:
        model = Sequential()
        # Handle both old 'int' config and new 'list' config for backward compatibility
        hidden_units = cfg.get("hidden_units", [64, 32])
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units] * cfg.get("num_layers", 2)

        dropout_rate = cfg.get("dropout", 0.2)

        for i, units in enumerate(hidden_units):
            # The last LSTM layer must have return_sequences=False
            is_last_layer = (i == len(hidden_units) - 1)

            model.add(
                LSTM(
                    units=units,
                    return_sequences=not is_last_layer,
                    input_shape=input_shape if i == 0 else None,
                )
            )
            model.add(Dropout(dropout_rate))

        model.add(Dense(1))

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.get("learning_rate", 0.0005))
        model.compile(optimizer=optimizer, loss="mse")

        logger.info(f"LSTM built with layers: {hidden_units}")
        return model
    except Exception as e:
        raise RuntimeError(f"LSTM model building failed: {e}")


def train_lstm(model: Sequential, X_train, y_train, cfg: Dict) -> Tuple[Sequential, Any]:
    """
    Train LSTM model with dynamic hyperparameters.
    """
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=6, min_lr=1e-6)
    ]

    # Time-series validation split (manual to avoid shuffling)
    val_split = 0.1
    split_at = int(len(X_train) * (1 - val_split))

    X_t, X_v = X_train[:split_at], X_train[split_at:]
    y_t, y_v = y_train[:split_at], y_train[split_at:]

    history = model.fit(
        X_t, y_t,
        epochs=cfg.get("epochs", 50),
        batch_size=cfg.get("batch_size", 32),
        validation_data=(X_v, y_v),
        callbacks=callbacks,
        verbose=1
    )

    return model, history
