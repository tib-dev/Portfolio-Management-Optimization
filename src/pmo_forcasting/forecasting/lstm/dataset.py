import numpy as np


def create_sequences(series, window_size):
    """
    Convert a time series into supervised learning sequences.

    Parameters
    ----------
    series : array-like
        Input time series.
    window_size : int
        Number of past observations per input sample.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Feature sequences X and target values y.
    """
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])
        y.append(series[i])
    return np.array(X), np.array(y)
