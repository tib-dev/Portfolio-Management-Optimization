import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, Tuple


def prepare_forecasting_data(df: pd.DataFrame, cfg: Dict) -> Dict[str, Any]:
    """
    Improved data preparation module that supports both ARIMA (raw data) 
    and LSTM (scaled sequences).
    """
    data_cfg = cfg["forecasting"]["data"]
    lstm_cfg = cfg["forecasting"]["lstm"]
    target = data_cfg["target_col"]
    date_col = data_cfg["date_col"]
    window = lstm_cfg.get("window_size", 60)

    # 1. Clean and Sort
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    # Remove timezone info for compatibility with ARIMA/LSTM
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # 2. Time-Based Split (Raw for ARIMA)
    train_df = df.loc[data_cfg["train_start"]:data_cfg["train_end"]]
    test_df = df.loc[data_cfg["test_start"]:data_cfg["test_end"]]

    # 3. Scaling (Mandatory for LSTM)
    # Fit ONLY on training data to prevent future information leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df[[target]])
    test_scaled = scaler.transform(test_df[[target]])

    # 4. Generate LSTM Sequences
    # We include a bit of the end of the train set for the first test sequence
    # to avoid losing the first 60 days of the test period.
    full_test_input = np.vstack((train_scaled[-window:], test_scaled))

    X_train, y_train = make_lstm_sequences(train_scaled, window)
    X_test, y_test = make_lstm_sequences(full_test_input, window)

    return {
        # Data for ARIMA (Needs raw dollar values)
        "y_train_raw": train_df[target],
        "y_test_raw": test_df[target],

        # Data for LSTM (Needs 3D scaled sequences)
        "X_train_lstm": X_train,
        "y_train_lstm": y_train,
        "X_test_lstm": X_test,
        "y_test_lstm": y_test,

        # Utilities for Evaluation & Plotting
        "scaler": scaler,
        "test_index": test_df.index,
        "target_col": target
    }


def make_lstm_sequences(data: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts 1D time series data into 3D sequences (Samples, Window, Features).
    """
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i, 0])
        y.append(data[i, 0])

    X_array = np.array(X).reshape(-1, window, 1)
    y_array = np.array(y)
    return X_array, y_array
