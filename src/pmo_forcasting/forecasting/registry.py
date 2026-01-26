import logging
from typing import Dict, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Lightweight in-memory model registry.

    Purpose
    -------
    Acts as a central place to register, track, and retrieve trained models
    along with their metadata and evaluation metrics.

    This is intentionally simple and transparent:
    - No serialization assumptions
    - No framework coupling
    - Easy to extend later to disk / MLflow / DB

    Typical usage
    -------------
    registry = ModelRegistry()

    registry.register(
        name="LSTM_TSLA",
        model=lstm_model,
        metadata={"type": "LSTM", "window": 60}
    )

    registry.update_metrics(
        name="LSTM_TSLA",
        metrics={"MAE": 13.2, "RMSE": 16.7}
    )

    best = registry.get_best(metric="RMSE", minimize=True)
    """

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._metrics: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        model: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a trained model.

        Parameters
        ----------
        name : str
            Unique model identifier (e.g., "ARIMA_TSLA").
        model : Any
            Trained model object.
        metadata : dict, optional
            Model-related information such as:
            - model type
            - hyperparameters
            - ticker
            - training window
        """
        if name in self._models:
            logger.warning("Overwriting existing model: %s", name)

        self._models[name] = model
        self._metadata[name] = metadata or {}
        self._metrics[name] = {}

        logger.info("Model registered: %s", name)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def update_metrics(self, name: str, metrics: Dict[str, float]) -> None:
        """
        Attach evaluation metrics to a registered model.

        Parameters
        ----------
        name : str
            Model identifier.
        metrics : dict
            Dictionary of metrics (MAE, RMSE, MAPE, etc.).
        """
        self._assert_exists(name)

        self._metrics[name].update(metrics)
        logger.info("Metrics updated for model: %s", name)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str) -> Any:
        """
        Retrieve a registered model by name.
        """
        self._assert_exists(name)
        return self._models[name]

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """
        Retrieve metadata for a model.
        """
        self._assert_exists(name)
        return self._metadata[name]

    def get_metrics(self, name: str) -> Dict[str, float]:
        """
        Retrieve evaluation metrics for a model.
        """
        self._assert_exists(name)
        return self._metrics[name]

    # ------------------------------------------------------------------
    # Model Selection
    # ------------------------------------------------------------------

    def get_best(
        self,
        metric: str,
        minimize: bool = True,
    ) -> Dict[str, Any]:
        """
        Select the best model based on a metric.

        Parameters
        ----------
        metric : str
            Metric name (e.g., "RMSE", "MAE").
        minimize : bool, default=True
            Whether lower is better (True for error metrics).

        Returns
        -------
        dict
            Dictionary with model name, model object, metadata, and metrics.
        """
        if not self._metrics:
            raise ValueError("No metrics available for model selection")

        scores = {
            name: m.get(metric)
            for name, m in self._metrics.items()
            if metric in m
        }

        if not scores:
            raise ValueError(f"No models contain metric '{metric}'")

        best_name = (
            min(scores, key=scores.get)
            if minimize
            else max(scores, key=scores.get)
        )

        logger.info("Best model by %s: %s", metric, best_name)

        return {
            "name": best_name,
            "model": self._models[best_name],
            "metadata": self._metadata[best_name],
            "metrics": self._metrics[best_name],
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """
        Return a tabular summary of all registered models.
        """
        rows = []

        for name in self._models:
            row = {"model": name}
            row.update(self._metadata.get(name, {}))
            row.update(self._metrics.get(name, {}))
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _assert_exists(self, name: str) -> None:
        if name not in self._models:
            raise KeyError(f"Model not found in registry: {name}")
