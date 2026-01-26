from pmdarima import auto_arima


def build_arima(y_train, cfg):
    """
    Build an ARIMA or SARIMA model using auto_arima.

    Parameters
    ----------
    y_train : array-like
        Training time series values.
    cfg : dict
        Configuration dictionary with ARIMA parameters.

    Returns
    -------
    pmdarima.ARIMA
        Configured ARIMA/SARIMA model.

    Raises
    ------
    RuntimeError
        If model construction fails.
    """
    try:
        model = auto_arima(
            y_train,
            seasonal=cfg["arima"]["seasonal"],
            m=cfg["arima"]["m"],
            start_p=cfg["arima"]["auto_arima"]["start_p"],
            start_q=cfg["arima"]["auto_arima"]["start_q"],
            max_p=cfg["arima"]["auto_arima"]["max_p"],
            max_q=cfg["arima"]["auto_arima"]["max_q"],
            trace=cfg["arima"]["auto_arima"]["trace"],
            error_action="ignore",
            suppress_warnings=True
        )
        return model
    except Exception as e:
        raise RuntimeError(f"ARIMA model build failed: {e}")
