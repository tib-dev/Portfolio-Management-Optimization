import logging
import pandas as pd
from pmo_forcasting.data.source import YahooFinanceProvider
from pmo_forcasting.data.formater import DataTransformer
from pmo_forcasting.data.persistance import StorageRepository
from typing import Dict

logger = logging.getLogger(__name__)

# ---  (The Orchestrator) ---
class MarketDataService:
    """Orchestrates the flow between Provider, Transformer, and Storage."""
    def __init__(self, provider: YahooFinanceProvider, repository: StorageRepository):
        self.provider = provider
        self.repository = repository
        self.transformer = DataTransformer()

    def run(self, config: Dict, force: bool = False) -> pd.DataFrame:
        assets = config["assets"]
        tickers = [a["ticker"] for a in assets]
        dr = config["date_range"]

        # Cache Check
        if not force and all(self.repository.exists(t) for t in tickers):
            logger.info("Using cached data.")
            return pd.concat([self.repository.load(t) for t in tickers])

        # Fetch and Process
        raw_data = self.provider.fetch(tickers, str(dr["start"]), str(dr["end"]))

        final_frames = []
        for asset in assets:
            clean_df = self.transformer.clean(raw_data, asset["ticker"], asset)
            self.repository.save(clean_df, asset["ticker"])
            final_frames.append(clean_df)

        return pd.concat(final_frames).sort_values(["date", "ticker"])
