import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, Tuple

def prepare_data_from_df(df: pd.DataFrame, config: Dict) -> Dict[str, Any]:
    """Processes an in-memory DataFrame with timezone handling."""
    try:
        cfg = config['forecasting']['data']
        target = cfg['target_col']
        date_col = cfg['date_col']

        # 1. Preprocessing & Timezone Normalization
        df_proc = df.copy()
        df_proc[date_col] = pd.to_datetime(df_proc[date_col])
        df_proc = df_proc.sort_values(date_col).set_index(date_col)

        # Check for timezone awareness and normalize to naive for consistent slicing
        if df_proc.index.tz is not None:
            df_proc.index = df_proc.index.tz_localize(None)

        # 2. Temporal Splitting
        train_df = df_proc.loc[cfg['train_start']:cfg['train_end']]
        test_df = df_proc.loc[cfg['test_start']:cfg['test_end']]

        # 3. Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_df[[target]])
        test_scaled = scaler.transform(test_df[[target]])

        return {
            "train_df": train_df,
            "test_df": test_df,
            "train_scaled": train_scaled,
            "test_scaled": test_scaled,
            "scaler": scaler,
            "target_col": target
        }
    except Exception as e:
        print(f"Data preparation failed: {e}")
        raise

def create_sequences(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Converts scaled array into 3D LSTM sequences."""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    X = np.array(X).reshape(-1, window_size, 1)
    return X, np.array(y)

def get_model_ready_data(df: pd.DataFrame, config: Dict) -> Dict[str, Any]:
    """Wrapper to generate ARIMA and LSTM formats from df."""
    prep = prepare_data_from_df(df, config)
    win_size = config['forecasting']['lstm']['window_size']

    X_train, y_train = create_sequences(prep["train_scaled"], win_size)
    X_test, y_test = create_sequences(prep["test_scaled"], win_size)

    return {
        "arima_train": prep["train_df"][prep["target_col"]],
        "arima_test": prep["test_df"][prep["target_col"]],
        "lstm_train": (X_train, y_train),
        "lstm_test": (X_test, y_test),
        "scaler": prep["scaler"]
    }

