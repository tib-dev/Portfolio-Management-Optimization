def forecast_arima(model, n_periods):
    """
    Generate out-of-sample forecasts using ARIMA.

    Parameters
    ----------
    model : pmdarima.ARIMA
        Trained ARIMA model.
    n_periods : int
        Number of future time steps to forecast.

    Returns
    -------
    numpy.ndarray
        Forecasted values.
    """
    try:
        return model.predict(n_periods=n_periods)
    except Exception as e:
        raise RuntimeError(f"ARIMA forecasting failed: {e}")
