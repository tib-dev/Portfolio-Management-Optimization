import pandas as pd
import yfinance as yf
import logging
from typing import  List
logger = logging.getLogger(__name__)

# ---  THE PROVIDER (External API Logic) ---
class YahooFinanceProvider:
    """Handles only the communication with yfinance."""
    def fetch(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        df = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=False,
            group_by='ticker',
            progress=False
        )
        if df.empty:
            raise RuntimeError("API returned no data.")
        df.index = pd.to_datetime(df.index, utc=True)
        return df


