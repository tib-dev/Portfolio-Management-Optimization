from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_lstm(input_shape, cfg):
    """
    Build an LSTM neural network for time series forecasting.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input sequence (timesteps, features).
    cfg : dict
        Configuration dictionary with LSTM hyperparameters.

    Returns
    -------
    tensorflow.keras.Model
        Compiled LSTM model.
    """
    model = Sequential()

    for i in range(cfg["lstm"]["num_layers"]):
        return_sequences = i < cfg["lstm"]["num_layers"] - 1
        model.add(
            LSTM(
                cfg["lstm"]["hidden_units"],
                return_sequences=return_sequences,
                input_shape=input_shape if i == 0 else None
            )
        )
        model.add(Dropout(cfg["lstm"]["dropout"]))

    model.add(Dense(1))
    model.compile(
        optimizer=Adam(learning_rate=cfg["lstm"]["learning_rate"]),
        loss="mse"
    )
    return model
