import joblib
from pathlib import Path
from typing import Dict, Any, List
import json
import logging

try:
    from tensorflow.keras.models import Model as KerasModel
except ImportError:
    KerasModel = ()

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Framework-aware model registry supporting:
    - Keras / TensorFlow
    - ARIMA / statsmodels
    - sklearn-style models
    """

    def __init__(self, base_dir="runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)

        self._runs: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        run_id: str,
        model: Any,
        metrics: Dict[str, float],
        config: Dict[str, Any],
    ):
        run_key = f"{name}::{run_id}"
        run_path = self.base_dir / f"{name}_{run_id}"
        run_path.mkdir(exist_ok=True)

        # -----------------------------
        # Save model (explicit typing)
        # -----------------------------
        if isinstance(model, KerasModel):
            framework = "keras"
            model_file = "model.keras"
            model.save(run_path / model_file)
        else:
            framework = "pickle"
            model_file = "model.pkl"
            joblib.dump(model, run_path / model_file)

        # Save metadata
        (run_path / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (run_path / "config.json").write_text(json.dumps(config, indent=2))

        # Registry entry
        self._runs[run_key] = {
            "name": name,
            "run_id": run_id,
            "framework": framework,
            "model_file": model_file,
            "path": run_path,
            "metrics": metrics,
            "config": config,
            "model": model,  # optional in-memory
        }

        logger.info("Registered %s model: %s", framework, run_key)

    def summary(self) -> List[Dict[str, Any]]:
        rows = []

        for r in self._runs.values():
            row = {
                "name": r["name"],
                "run_id": r["run_id"],
                "framework": r["framework"],
                "path": r["path"],
                "model_file": r["path"] / r["model_file"],
                "config": r["config"],
                "metrics": r["metrics"],
            }
            row.update(r["metrics"])  # flatten metrics
            rows.append(row)

        return rows
