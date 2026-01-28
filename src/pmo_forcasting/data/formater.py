import pandas as pd
import logging
from typing import Dict
logger = logging.getLogger(__name__)


# --- THE TRANSFORMER
class DataTransformer:
    """Handles only formatting and metadata application."""
    @staticmethod
    def clean(raw_df: pd.DataFrame, ticker: str, asset_info: Dict) -> pd.DataFrame:
        try:
            # Extract ticker slice
            df = raw_df[ticker].copy() if isinstance(raw_df.columns, pd.MultiIndex) else raw_df.copy()

            # Formatting
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df = df.reset_index().rename(columns={df.index.name: "date"})

            # Metadata
            df["ticker"] = ticker
            df["asset_class"] = asset_info.get("asset_class", "unknown")
            df["risk_profile"] = asset_info.get("risk_profile", "unknown")

            return df.dropna(subset=["close"]).reset_index(drop=True)
        except Exception as e:
            logger.error(f"Transformation failed for {ticker}: {e}")
            return pd.DataFrame()
