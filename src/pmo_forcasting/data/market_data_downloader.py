import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketDataDownloader:
    """
    Downloads and aligns OHLCV data for multiple assets.
    Output columns: date, open, high, low, close, adj_close, volume, 
                    ticker, asset_class, risk_profile
    """

    def __init__(self, config: Dict, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)

    def run(self, force: bool = False) -> pd.DataFrame:
        tickers = [a["ticker"] for a in self.config["assets"]]
        csv_paths = {t: self.output_dir / f"{t}.csv" for t in tickers}

        if all(p.exists() for p in csv_paths.values()) and not force:
            logger.info("Loading cached CSVs")
            frames = [self._load_csv(p) for p in csv_paths.values()]
            return self._combine(frames)

        logger.info("Downloading batch data for %s", tickers)
        raw_df = self._download_raw(tickers)

        frames = []
        for asset in self.config["assets"]:
            df = self._extract_and_format(raw_df, asset)
            if not df.empty:
                self._save_csv(df, csv_paths[asset["ticker"]])
                frames.append(df)

        return self._combine(frames)

    def _download_raw(self, tickers: List[str]) -> pd.DataFrame:
        dr = self.config["date_range"]

        # Download all fields at once
        df = yf.download(
            tickers=tickers,
            start=str(dr["start"]),
            end=str(dr["end"]),
            auto_adjust=False,
            progress=False,
            threads=True
        )

        if df is None or df.empty:
            raise RuntimeError("Yahoo Finance returned no data.")

        # Standardize Index
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    def _extract_and_format(self, raw_df: pd.DataFrame, asset: Dict) -> pd.DataFrame:
        ticker = asset["ticker"]

        # Handle MultiIndex (multiple tickers) vs Single Index (one ticker)
        try:
            if isinstance(raw_df.columns, pd.MultiIndex):
                # Extract all OHLVC for this specific ticker
                df = raw_df.xs(ticker, axis=1, level=1).copy()
            else:
                df = raw_df.copy()

            # Normalize column names to lowercase and handle 'Adj Close'
            df.columns = df.columns.str.lower()
            if "adj close" in df.columns:
                df = df.rename(columns={"adj close": "adj_close"})

            # Drop rows where Close is missing (to align calendars)
            df = df.dropna(subset=["close"])

            # Reset index and fix the 'date' column name
            df = df.reset_index()
            df.columns = ["date" if c.lower() in ["date", "index"]
                          else c for c in df.columns]

            # Add Metadata
            df["ticker"] = ticker
            df["asset_class"] = asset.get("asset_class", "unknown")
            df["risk_profile"] = asset.get("risk_profile", "unknown")

            # Final Column Ordering
            desired_cols = [
                "date", "open", "high", "low", "close", "adj_close",
                "volume", "ticker", "asset_class", "risk_profile"
            ]
            # Only keep columns that exist (in case volume is missing for some assets)
            existing_cols = [c for c in desired_cols if c in df.columns]

            return df[existing_cols].sort_values("date").reset_index(drop=True)

        except Exception as e:
            logger.error("Error extracting %s: %s", ticker, e)
            return pd.DataFrame()

    def _combine(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])

    def _save_csv(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def _load_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df
