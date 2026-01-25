# preprocessing/outliers.py
import pandas as pd
import numpy as np


def detect_outliers(df: pd.DataFrame, returns_col: str = "returns",
                    method: str = "zscore", threshold: float = 3.0) -> pd.DataFrame:
    """
    Flag outliers in daily returns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a returns column
    returns_col : str
        Name of column containing daily returns
    method : str
        'zscore' or 'percentile'
    threshold : float
        Z-score threshold or percentile cutoff

    Returns
    -------
    pd.DataFrame
        Original df with boolean 'outlier' column
    """
    df = df.copy()

    if method == "zscore":
        mean = df[returns_col].mean()
        std = df[returns_col].std()
        df["outlier"] = np.abs((df[returns_col] - mean) / std) > threshold
    elif method == "percentile":
        lower = df[returns_col].quantile((1 - threshold) / 2)
        upper = df[returns_col].quantile(1 - (1 - threshold) / 2)
        df["outlier"] = (df[returns_col] < lower) | (df[returns_col] > upper)
    else:
        raise ValueError("method must be 'zscore' or 'percentile'")

    return df
