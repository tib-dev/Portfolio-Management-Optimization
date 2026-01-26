def run_pipeline(df, config_path):
    """
    Run the end-to-end forecasting pipeline.

    This function orchestrates:
    - configuration loading
    - data splitting
    - ARIMA training and forecasting
    - LSTM training and forecasting
    - model evaluation and comparison

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed Tesla price data.
    config_path : str
        Path to forecasting YAML configuration.

    Returns
    -------
    dict
        Evaluation metrics for each model.
    """
    import pandas as pd
    from pmo_forcasting.core.settings import settings
    from pmo_forcasting.forecasting.splits import time_series_split
    from pmo_forcasting.forecasting.arima.model import build_arima
    from pmo_forcasting.forecasting.arima.forecast import forecast_arima
    from pmo_forcasting.forecasting.lstm.dataset import create_sequences
    from pmo_forcasting.forecasting.lstm.model import build_lstm
    from pmo_forcasting.forecasting.lstm.train import train_lstm
    from pmo_forcasting.forecasting.lstm.forecast import forecast_lstm
    from pmo_forcasting.forecasting.evaluate import evaluate

    cfg = settings.config
    train, test = time_series_split(df, cfg)
    y_train = train[cfg["forecasting"]["target_col"]]
    y_test = test[cfg["forecasting"]["target_col"]]

    # ARIMA
    arima = build_arima(y_train, cfg)
    arima_preds = forecast_arima(arima, len(y_test))
    arima_metrics = evaluate(y_test.values, arima_preds)

    # LSTM
    X_train, y_train_lstm = create_sequences(
        y_train.values, cfg["lstm"]["window_size"])
    combined = pd.concat([y_train, y_test])
    X_test, y_test_lstm = create_sequences(
        combined.values, cfg["lstm"]["window_size"])

    lstm = build_lstm((X_train.shape[1], 1), cfg)
    lstm, _ = train_lstm(lstm, X_train[..., None], y_train_lstm, cfg)
    lstm_preds = forecast_lstm(lstm, X_test[..., None])
    lstm_metrics = evaluate(y_test_lstm, lstm_preds)

    return {
        "ARIMA": arima_metrics,
        "LSTM": lstm_metrics
    }


class ForecastingPipeline:
    """
    Orchestrates the full time series forecasting lifecycle.

    This pipeline follows industry best practices for
    financial time series modeling:
    - chronological splitting
    - model-specific training logic
    - out-of-sample evaluation
    - reproducible configuration via YAML
    """

    def __init__(self, config, model_builder, trainer, forecaster, evaluator):
        """
        Parameters
        ----------
        config : dict
            Experiment configuration loaded from YAML.
        model_builder : callable
            Function that constructs a model.
        trainer : callable
            Function that trains the model.
        forecaster : callable
            Function that generates forecasts.
        evaluator : callable
            Function that computes evaluation metrics.
        """
        self.config = config
        self.model_builder = model_builder
        self.trainer = trainer
        self.forecaster = forecaster
        self.evaluator = evaluator

    def run(self, train_data, test_data):
        """
        Execute the forecasting pipeline.

        Returns
        -------
        dict
            Evaluation metrics.
        """
        model = self.model_builder(train_data, self.config)
        model = self.trainer(model)
        preds = self.forecaster(model, len(test_data))
        return self.evaluator(test_data, preds)
