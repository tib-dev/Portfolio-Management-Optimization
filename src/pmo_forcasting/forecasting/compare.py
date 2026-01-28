"""
forecasting/compare.py

Model comparison, selection, and champion model persistence.
"""

from pmo_forcasting.core.project_root import get_project_root
from typing import Dict, Any
import pandas as pd
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BEST_MODEL_DIR = get_project_root() / "models" / "best_models"
BEST_MODEL_DIR.mkdir(exist_ok=True, parents=True)


class ModelComparator:
    """
    Compare registered forecasting models and select the best one
    based on a priority metric (e.g., RMSE, MAE, MAPE).
    """

    def __init__(self, metrics_priority: str = "RMSE", minimize: bool = True):
        self.metric = metrics_priority
        self.minimize = minimize

    def compare(self, registry) -> pd.DataFrame:
        """
        Build a comparison table from the ModelRegistry.

        Returns
        -------
        pd.DataFrame
            Sorted comparison table.
        """
        raw = registry.summary()

        # Registry returns list[dict] → normalize immediately
        df = pd.DataFrame(raw)

        if df.empty:
            raise ValueError(
                "No models available in registry. Train and register models first."
            )

        if self.metric not in df.columns:
            raise KeyError(
                f"Metric '{self.metric}' not found. "
                f"Available metrics: {list(df.columns)}"
            )

        logger.info("Model comparison table created")
        return df.sort_values(self.metric, ascending=self.minimize).reset_index(drop=True)

    def select_best(self, registry) -> Dict[str, Any]:
        df = self.compare(registry)
        best_row = df.iloc[0]

        # 1. Clean up the naming to avoid duplicated timestamps
        best_name = str(best_row["name"])
        run_id = str(best_row["run_id"])

        # Only append run_id if it's not already part of the name
        folder_name = best_name if run_id in best_name else f"{best_name}_{run_id}"
        dest_path = BEST_MODEL_DIR / folder_name

        best = {
            "name": best_name,
            "run_id": run_id,
            "metrics": best_row.get("metrics", {}),
            "config": best_row.get("config", {}),
            "source_path": Path(best_row["path"]),
        }

        # 2. Safety Check: Verify source exists before attempting copy
        if not best["source_path"].exists():
            # Debugging: check if the path is relative or absolute
            logger.error(f"Source not found: {best['source_path'].absolute()}")
            raise FileNotFoundError(
                f"Registered model path does not exist: {best['source_path']}")

        # 3. Handle destination (Clean it if it exists)
        if dest_path.exists():
            shutil.rmtree(dest_path)

        # 4. Copy the entire folder
        shutil.copytree(best["source_path"], dest_path)
