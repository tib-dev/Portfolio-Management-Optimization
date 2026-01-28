import pandas as pd
from pathlib import Path
import logging
logger = logging.getLogger(__name__)



# ---  THE REPOSITORY (Storage Logic) ---
class StorageRepository:
    """Handles only reading and writing to the disk."""
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, df: pd.DataFrame, ticker: str):
        path = self.base_path / f"{ticker}.csv"
        df.to_csv(path, index=False)

    def load(self, ticker: str) -> pd.DataFrame:
        path = self.base_path / f"{ticker}.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    def exists(self, ticker: str) -> bool:
        return (self.base_path / f"{ticker}.csv").exists()

