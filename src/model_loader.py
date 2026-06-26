"""Model loading and prediction utilities."""
import logging

import joblib
import numpy as np

from src.api.config import MODEL_PATH

logger = logging.getLogger(__name__)
_MODEL = None


def load() -> None:
    """Load the trained model from disk."""
    global _MODEL
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}.")
    _MODEL = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)


def is_model_loaded() -> bool:
    """Check if model is currently loaded."""
    return _MODEL is not None


def predict(features: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Generate prediction for input features.

    Args:
        features: Input feature array

    Returns:
        Tuple of (predicted_class, probabilities)
    """
    if _MODEL is None:
        raise RuntimeError("Model is not loaded.")
    return int(_MODEL.predict(features)[0]), _MODEL.predict_proba(features)[0]
