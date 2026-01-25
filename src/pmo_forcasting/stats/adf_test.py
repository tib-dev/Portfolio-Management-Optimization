
from typing import Dict
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import logging

logger = logging.getLogger(__name__)


class ADFTester:
    """
    Perform Augmented Dickey-Fuller (ADF) test for stationarity
    on closing prices and daily returns for multiple tickers.
    """

    def __init__(self, df: pd.DataFrame, ticker_col: str = "ticker"):
        """
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with at least columns ['date', 'close', 'returns', ticker_col]
        ticker_col : str
            Column name for asset ticker/identifier
        """
        self.df = df.copy()
        self.ticker_col = ticker_col

    def run(self) -> pd.DataFrame:
        """
        Run ADF test per ticker for closing prices and returns.

        Returns
        -------
        pd.DataFrame
            Tickers as index, columns:
            ['price_adf_stat', 'price_p_value', 'price_lags', 'price_obs',
             'returns_adf_stat', 'returns_p_value', 'returns_lags', 'returns_obs']
        """
        results: Dict[str, Dict] = {}

        for ticker in self.df[self.ticker_col].unique():
            subset = self.df[self.df[self.ticker_col]
                             == ticker].sort_values("date")

            try:
                # Closing price ADF
                price_series = subset["close"]
                price_adf = adfuller(price_series, autolag="AIC")

                # Returns ADF
                returns_series = subset["daily_return"].dropna()
                returns_adf = adfuller(returns_series, autolag="AIC")

                results[ticker] = {
                    "price_adf_stat": price_adf[0],
                    "price_p_value": price_adf[1],
                    "price_lags": price_adf[2],
                    "price_obs": price_adf[3],
                    "returns_adf_stat": returns_adf[0],
                    "returns_p_value": returns_adf[1],
                    "returns_lags": returns_adf[2],
                    "returns_obs": returns_adf[3],
                }

            except Exception as exc:
                logger.exception(
                    "ADF test failed for ticker %s: %s", ticker, exc)
                continue

        return pd.DataFrame(results).T
