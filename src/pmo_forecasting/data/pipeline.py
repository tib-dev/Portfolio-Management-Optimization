import logging
import pandas as pd
from typing import Dict, List

from pmo_forecasting.data.source import YahooFinanceProvider
from pmo_forecasting.data.transformer import DataTransformer
from pmo_forecasting.data.persistence import StorageRepository

logger = logging.getLogger(__name__)


class MarketDataService:
    """
    Orchestrates Provider → Transformer → Repository.
    """

    def __init__(
        self,
        provider: YahooFinanceProvider,
        repository: StorageRepository,
    ):
        self.provider = provider
        self.repository = repository
        self.transformer = DataTransformer()

    def run(self, config: Dict, force: bool = False) -> pd.DataFrame:
        assets = config["assets"]
        tickers: List[str] = [a["ticker"] for a in assets]
        date_range = config["date_range"]

        # ---------- Cache check ----------
        if not force and all(self.repository.exists(t) for t in tickers):
            logger.info("Using cached market data")
            frames = [self.repository.load(t) for t in tickers]
            return pd.concat(frames, ignore_index=True)

        # ---------- Fetch ----------
        raw_data = self.provider.fetch(
            tickers=tickers,
            start=str(date_range["start"]),
            end=str(date_range["end"]),
        )

        # ---------- Transform + Persist ----------
        cleaned_frames = []
        for asset in assets:
            df = self.transformer.clean(
                raw_df=raw_data,
                ticker=asset["ticker"],
                asset_info=asset,
            )
            self.repository.save(df, asset["ticker"])
            cleaned_frames.append(df)

        return (
            pd.concat(cleaned_frames, ignore_index=True)
            .sort_values(["date", "ticker"])
        )
