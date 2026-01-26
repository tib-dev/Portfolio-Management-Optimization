def train_lstm(model, X_train, y_train, cfg):
    """
    Train an LSTM model on time series sequences.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Compiled LSTM model.
    X_train : numpy.ndarray
        Training input sequences.
    y_train : numpy.ndarray
        Training targets.
    cfg : dict
        Training hyperparameters.

    Returns
    -------
    (tensorflow.keras.Model, History)
        Trained model and training history.
    """
    try:
        history = model.fit(
            X_train,
            y_train,
            epochs=cfg["lstm"]["epochs"],
            batch_size=cfg["lstm"]["batch_size"],
            verbose=1
        )
        return model, history
    except Exception as e:
        raise RuntimeError(f"LSTM training failed: {e}")
