def train_arima(model):
    """
    Fit an ARIMA/SARIMA model on training data.

    Parameters
    ----------
    model : pmdarima.ARIMA
        Unfitted ARIMA model.

    Returns
    -------
    pmdarima.ARIMA
        Trained ARIMA model.
    """
    try:
        model.fit(model.y_)
        return model
    except Exception as e:
        raise RuntimeError(f"ARIMA training failed: {e}")
