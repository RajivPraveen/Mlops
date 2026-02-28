import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "model" / "wine_model.pkl"


def predict_data(X):
    """
    Predict the wine cultivar class for the input features.

    Args:
        X (array-like): Input features of shape (n_samples, 13).

    Returns:
        numpy.ndarray: Predicted class labels.
    """
    model = joblib.load(MODEL_PATH)
    return model.predict(X)


def predict_proba(X):
    """
    Return class probabilities for the input features.

    Args:
        X (array-like): Input features of shape (n_samples, 13).

    Returns:
        numpy.ndarray: Class probabilities of shape (n_samples, 3).
    """
    model = joblib.load(MODEL_PATH)
    return model.predict_proba(X)
