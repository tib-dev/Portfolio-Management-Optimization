
from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_financial_data(df: pd.DataFrame,
                              adjust_prices: bool = True,
                              fill_method: str = "ffill",
                              drop_non_trading: bool = True,
                              compute_returns: bool = True,
                              scale: bool = False) -> pd.DataFrame:
    df = df.copy()

    # Adjust prices
    if adjust_prices and "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    # Fill missing
    if fill_method == "ffill":
        df = df.sort_values(["ticker", "date"]).groupby("ticker").ffill()
    elif fill_method == "bfill":
        df = df.sort_values(["ticker", "date"]).groupby("ticker").bfill()

    # Drop non-trading
    if drop_non_trading:
        price_cols = [c for c in ["open", "high", "low",
                                  "close", "adj_close"] if c in df.columns]
        df = df.dropna(subset=price_cols, how="all")

    # Calculate daily returns
    if compute_returns:
        df["returns"] = df.groupby("ticker")["close"].pct_change()

    # Scale (optional)
    if scale:
        scaler = StandardScaler()
        numeric_cols = ["open", "high", "low",
                        "close", "adj_close", "volume", "returns"]
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
