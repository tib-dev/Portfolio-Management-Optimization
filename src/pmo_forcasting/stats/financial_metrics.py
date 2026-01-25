# stats/financial_metrics.py
from typing import Dict
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FinancialMetrics:
    """
    Compute key risk and return metrics for financial time series data.

    Metrics:
    - Daily mean return
    - Annualized volatility
    - Cumulative return
    - Sharpe ratio (assumes risk-free rate = 0)
    - Value at Risk (historical, 5% quantile)
    """

    def __init__(self, df: pd.DataFrame, ticker_col: str = "ticker"):
        """
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with at least ['daily_return', ticker_col] columns
        ticker_col : str
            Column name representing ticker or asset
        """
        self.df = df.copy()
        self.ticker_col = ticker_col

        if "daily_return" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'daily_return' column")

    def run(self) -> pd.DataFrame:
        """
        Compute metrics per ticker.

        Returns
        -------
        pd.DataFrame
            Metrics indexed by ticker
        """
        metrics: Dict[str, Dict] = {}

        for ticker in self.df[self.ticker_col].unique():
            subset = self.df[self.df[self.ticker_col]
                             == ticker].sort_values("date")
            returns = subset["daily_return"].dropna()

            if returns.empty:
                logger.warning(
                    "No returns data for %s, skipping metrics", ticker)
                continue

            try:
                # Basic metrics
                mean_daily = returns.mean()
                annual_vol = returns.std() * np.sqrt(252)  # trading days
                cumulative = (1 + returns).prod() - 1
                sharpe = mean_daily / returns.std() * np.sqrt(252)
                var_5 = returns.quantile(0.05)

                metrics[ticker] = {
                    "mean_daily_return": mean_daily,
                    "annualized_volatility": annual_vol,
                    "cumulative_return": cumulative,
                    "sharpe_ratio": sharpe,
                    "VaR_5": var_5,
                }

            except Exception as exc:
                logger.exception(
                    "Metrics calculation failed for %s: %s", ticker, exc)
                continue

        return pd.DataFrame(metrics).T
