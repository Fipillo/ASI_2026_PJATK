"""Model evaluation nodes for NBA pipeline."""
import os
import warnings
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def _configure_mlflow(mlflow_tracking_uri: str | None, mlflow_experiment_name: str | None) -> None:
    if mlflow_tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    if mlflow_experiment_name:
        try:
            mlflow.set_experiment(mlflow_experiment_name)
        except Exception:
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            mlflow.set_tracking_uri(mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
            mlflow.set_experiment(mlflow_experiment_name)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    mlflow_tracking_uri: str = None,
    mlflow_experiment_name: str = None,
) -> Dict[str, float]:
    """
    Evaluate trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Handle both numpy arrays and pandas DataFrames from predict_proba
    if isinstance(y_proba, pd.DataFrame):
        y_proba = y_proba.iloc[:, 1].values
    elif isinstance(y_proba, np.ndarray):
        y_proba = y_proba[:, 1]
    else:
        # Fallback: try to convert to numpy
        y_proba = np.array(y_proba)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "test_rows": len(X_test),
        "features_count": X_test.shape[1],
    }

    if _configure_mlflow(mlflow_tracking_uri, mlflow_experiment_name):
        # Log metrics to MLflow
        with mlflow.start_run(run_name="baseline_evaluation"):
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
    else:
        warnings.warn("Skipping evaluation MLflow logging due to tracking backend issues.", UserWarning)

    return metrics
