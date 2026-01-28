import logging
import pandas as pd
from typing import Dict

logger = logging.getLogger(__name__)


class DataTransformer:
    """
    Cleans raw provider data and applies asset metadata.
    """

    @staticmethod
    def clean(
        raw_df: pd.DataFrame,
        ticker: str,
        asset_info: Dict,
    ) -> pd.DataFrame:
        try:
            # Handle multi-index from yfinance
            if isinstance(raw_df.columns, pd.MultiIndex):
                df = raw_df[ticker].copy()
            else:
                df = raw_df.copy()

            # Normalize columns
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            # Reset index → date column
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "date"}, inplace=True)

            # Normalize datetime (THIS fixes your earlier errors)
            df["date"] = (
                pd.to_datetime(df["date"], utc=True)
                .dt.normalize()
            )

            # Metadata
            df["ticker"] = ticker
            df["asset_class"] = asset_info.get("asset_class", "unknown")
            df["risk_profile"] = asset_info.get("risk_profile", "unknown")

            return (
                df.dropna(subset=["close"])
                  .reset_index(drop=True)
            )

        except Exception as exc:
            logger.exception("Transformation failed for %s", ticker)
            raise RuntimeError(f"Transformation failed for {ticker}: {exc}")
