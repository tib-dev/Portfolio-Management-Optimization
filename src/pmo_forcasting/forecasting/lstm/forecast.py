def forecast_lstm(model, X_test):
    """
    Generate forecasts using a trained LSTM model.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Trained LSTM model.
    X_test : numpy.ndarray
        Input sequences for forecasting.

    Returns
    -------
    numpy.ndarray
        Predicted values.
    """
    try:
        return model.predict(X_test).flatten()
    except Exception as e:
        raise RuntimeError(f"LSTM forecasting failed: {e}")
