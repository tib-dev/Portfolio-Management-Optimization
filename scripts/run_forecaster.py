import pandas as pd
import tensorflow as tf

from pmo_forcasting.core.settings import settings
from pmo_forcasting.data.handler import DataHandler
import pmo_forcasting.forecasting.data_preparation as dp
import pmo_forcasting.forecasting.lstm.recursive as fr


def main():
    # --------------------------------------------------
    # 1. Load trained LSTM model
    # --------------------------------------------------
    MODEL_PATH = (
        settings.paths.MODELS["models_dir"]
        / "best_models"
        / "model.keras"
    )

    lstm_model = tf.keras.models.load_model(MODEL_PATH)
    lstm_model.summary()

    # --------------------------------------------------
    # 2. Load and filter data
    # --------------------------------------------------
    df = DataHandler.from_registry(
        "DATA",
        "processed_dir",
        "processed_prices_data.csv"
    ).load()

    # Filter for TSLA only
    df = df[df["ticker"] == "TSLA"].copy()
    df.reset_index(drop=True, inplace=True)

    # --------------------------------------------------
    # 3. Prepare forecasting data
    # --------------------------------------------------
    config = settings.config
    prep = dp.prepare_forecasting_data(df, config)

    X_train_lstm = prep["X_train_lstm"]
    y_train_lstm = prep["y_train_lstm"]
    X_test_lstm = prep["X_test_lstm"]
    y_test_lstm = prep["y_test_lstm"]

    scaler = prep["scaler"]
    test_index = prep["test_index"]

    print("Data ready ✔")
    print("Train:", X_train_lstm.shape, "Test:", X_test_lstm.shape)

    # --------------------------------------------------
    # 4. Recursive multi-step LSTM forecast
    # --------------------------------------------------
    window_size = X_test_lstm.shape[1]
    last_window = X_test_lstm[-1].reshape(1, window_size, 1)

    N_FORECAST_DAYS = config["forecasting"]["lstm"]["forecasting_days"]

    forecast_scaled = fr.recursive_lstm_forecast(
        model=lstm_model,
        seed_window=last_window,
        n_steps=N_FORECAST_DAYS
    )

    forecast_prices = fr.inverse_scale_forecast(
        forecast_scaled,
        scaler=scaler
    )

    forecast_index = fr.build_forecast_index(
        last_date=test_index[-1],
        n_steps=N_FORECAST_DAYS
    )

    lstm_forecast = pd.Series(
        forecast_prices,
        index=forecast_index,
        name="TSLA_LSTM_Forecast"
    )

    # --------------------------------------------------
    # 5. Confidence intervals (from test residuals)
    # --------------------------------------------------
    y_test_true = scaler.inverse_transform(
        y_test_lstm.reshape(-1, 1)
    ).flatten()

    test_preds = scaler.inverse_transform(
        lstm_model.predict(X_test_lstm).reshape(-1, 1)
    ).flatten()

    residuals = y_test_true - test_preds

    lower_ci, upper_ci = fr.compute_confidence_intervals(
        forecast=lstm_forecast,
        residuals=residuals
    )

    # --------------------------------------------------
    # 6. Trend analysis
    # --------------------------------------------------
    pct_change = (
        (lstm_forecast.iloc[-1] / lstm_forecast.iloc[0]) - 1
    ) * 100

    trend = "Upward" if pct_change > 0 else "Downward"

    print(
        f"""
Trend Analysis
--------------
Direction: {trend}
Start Price: {lstm_forecast.iloc[0]:.2f}
End Price: {lstm_forecast.iloc[-1]:.2f}
Expected Change: {pct_change:.2f}%
"""
    )

    return {
        "forecast": lstm_forecast,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci
    }


if __name__ == "__main__":
    main()
