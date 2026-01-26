import os
import joblib


def register_model(model, name, cfg):
    """
    Persist a trained model to disk.

    Parameters
    ----------
    model : object
        Trained model object.
    name : str
        Model name identifier.
    cfg : dict
        Configuration dictionary with registry settings.

    Returns
    -------
    str
        Path where the model was saved.
    """
    try:
        os.makedirs(cfg["registry"]["save_dir"], exist_ok=True)
        path = os.path.join(cfg["registry"]["save_dir"], f"{name}.pkl")
        joblib.dump(model, path)
        return path
    except Exception as e:
        raise RuntimeError(f"Model registration failed: {e}")
