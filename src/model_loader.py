"""Model loading and prediction utilities."""
import logging

import joblib
import numpy as np

from src.api.config import MODEL_PATH

logger = logging.getLogger(__name__)
_model = None


def load() -> None:
    """Load the trained model from disk."""
    global _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}.")
    _model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)


def is_model_loaded() -> bool:
    """Check if model is currently loaded."""
    return _model is not None


def predict(features: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Generate prediction for input features.

    Args:
        features: Input feature array

    Returns:
        Tuple of (predicted_class, probabilities)
    """
    if _model is None:
        raise RuntimeError("Model is not loaded.")
    return int(_model.predict(features)[0]), _model.predict_proba(features)[0]
