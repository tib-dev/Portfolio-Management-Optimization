from datetime import datetime
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from pmo_forcasting.forecasting.arima.model import build_arima, train_arima
from pmo_forcasting.forecasting.arima.forecast import forecast_arima
from pmo_forcasting.forecasting.lstm.model import build_lstm, train_lstm
from pmo_forcasting.forecasting.lstm.forecast import forecast_lstm
from pmo_forcasting.forecasting.evaluate import evaluate
from pmo_forcasting.forecasting.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ForecastingPipeline:
    """
    End-to-end forecasting pipeline for time series models.

    Supported models
    ----------------
    - ARIMA / SARIMA (via pmdarima.auto_arima)
    - LSTM (Keras)

    Responsibilities
    ----------------
    - Train model
    - Generate forecasts
    - Evaluate with MAE / RMSE / MAPE
    - Register models and metrics with run-level isolation
    """

    def __init__(
        self,
        config: Dict[str, Any],
        registry: ModelRegistry,
        models: List[str],
    ):
        self.config = config
        self.registry = registry
        self.models = models

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, data_bundle: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Run forecasting for selected models.

        Parameters
        ----------
        data_bundle : dict
            Output of get_model_ready_data()

        Returns
        -------
        dict
            {model_name: metrics}
        """
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        logger.info("Starting forecasting run: %s", run_id)
        logger.info("Models requested: %s", self.models)

        results: Dict[str, Dict[str, float]] = {}

        for model_name in self.models:
            try:
                if model_name == "arima":
                    results["arima"] = self._run_arima(data_bundle, run_id)

                elif model_name == "lstm":
                    results["lstm"] = self._run_lstm(data_bundle, run_id)

                else:
                    logger.warning("Unknown model '%s' – skipped", model_name)

            except Exception as exc:
                logger.exception("Model '%s' failed", model_name)
                results[model_name] = {"error": str(exc)}

        return results

    # ------------------------------------------------------------------
    # ARIMA
    # ------------------------------------------------------------------

    def _run_arima(self, data: Dict[str, Any], run_id: str) -> Dict[str, float]:
        logger.info("Running ARIMA pipeline step")

        y_train: pd.Series = data["arima_train"].dropna()
        y_test: pd.Series = data["arima_test"].dropna()
        cfg = self.config["forecasting"]["arima"]

        # Safety check
        if y_train.index.max() >= y_test.index.min():
            raise ValueError("ARIMA train/test split overlaps in time")

        # Build + train
        model_spec = build_arima(y_train, cfg)
        model = train_arima(model_spec, y_train)

        # Forecast
        preds = forecast_arima(
            model,
            n_periods=len(y_test),
            index=y_test.index,
        )

        y_true = y_test.loc[preds.index]

        metrics = evaluate(y_true.values, preds.values)

        model_name = f"arima_{run_id}"
        self.registry.register(
            name=model_name,
            run_id=run_id,
            model=model,
            metrics=metrics,
            config=cfg,
        )

        logger.info("ARIMA finished. Metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # LSTM
    # ------------------------------------------------------------------


    def _run_lstm(self, data: Dict[str, Any], run_id: str) -> Dict[str, float]:
        logger.info("Running LSTM pipeline step")
    
        # Extract data from bundle
        X_train, y_train = data["lstm_train"]
        X_test, y_test = data["lstm_test"]
        scaler = data["scaler"]
        cfg = self.config["forecasting"]["lstm"]
    
        # 1. Build and Train
        model = build_lstm(input_shape=(
            X_train.shape[1], X_train.shape[2]), cfg=cfg)
        trained_model, _ = train_lstm(model, X_train, y_train, cfg)
    
        # 2. Generate Predictions (Result is 0-1 scaled)
        # Ensure your forecast_lstm doesn't internally inverse_scale yet for consistency
        preds_scaled = trained_model.predict(X_test)
    
        # 3. CRITICAL FIX: Inverse Scale both Predictions AND Ground Truth
        # Reshape is required for the scaler: (n_samples, 1)
        preds_dollars = scaler.inverse_transform(preds_scaled).flatten()
    
        y_true_scaled = y_test.reshape(-1, 1)
        y_true_dollars = scaler.inverse_transform(y_true_scaled).flatten()
    
        # 4. Evaluate USD vs USD
        # This will fix the 47,000% MAPE and $351 MAE
        metrics = evaluate(y_true_dollars, preds_dollars)
    
        # 5. Register
        model_name = f"lstm_{run_id}"
        self.registry.register(
            name=model_name,
            run_id=run_id,
            model=trained_model,
            metrics=metrics,
            config=cfg,
        )
    
        logger.info("LSTM finished. Corrected Metrics: %s", metrics)
        return metrics
