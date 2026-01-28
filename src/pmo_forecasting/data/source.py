import logging
import pandas as pd
import yfinance as yf
from typing import List

logger = logging.getLogger(__name__)


class YahooFinanceProvider:
    """
    Fetches raw market data from Yahoo Finance.
    """

    def fetch(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        df = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=False,
            group_by="ticker",
            progress=False,
        )

        if df.empty:
            raise RuntimeError("Yahoo Finance returned empty dataset")

        # Always enforce timezone-aware datetime index
        df.index = pd.to_datetime(df.index, utc=True)

        return df
