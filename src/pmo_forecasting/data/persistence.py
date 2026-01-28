import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StorageRepository:
    """
    Handles disk persistence (CSV-based).
    """

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str) -> Path:
        return self.base_path / f"{ticker}.csv"

    def save(self, df: pd.DataFrame, ticker: str) -> None:
        df.to_csv(self._path(ticker), index=False)

    def load(self, ticker: str) -> pd.DataFrame:
        path = self._path(ticker)
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    def exists(self, ticker: str) -> bool:
        return self._path(ticker).exists()
