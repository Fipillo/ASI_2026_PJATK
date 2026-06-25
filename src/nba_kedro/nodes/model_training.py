"""Model training nodes for NBA pipeline."""
import os
import warnings
from typing import Any, Dict, Tuple

import mlflow
import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


def _configure_mlflow(mlflow_tracking_uri: str | None, mlflow_experiment_name: str | None) -> bool:
    if mlflow_tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    try:
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        if mlflow_experiment_name:
            mlflow.set_experiment(mlflow_experiment_name)
        return True
    except Exception as exc:
        warnings.warn(
            f"MLflow configuration failed: {exc}. "
            "Continuing without MLflow logging.",
            UserWarning,
        )
        return False


def train_baseline_model(
    features: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    params: Dict[str, Any] = None,
    mlflow_tracking_uri: str = None,
    mlflow_experiment_name: str = None,
) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame]:
    """
    Train baseline Random Forest model with chronological split.
    
    Args:
        features: Feature dataset with target column
        test_size: Proportion of test data
        random_state: Random state for reproducibility
        params: Model parameters (optional)
        
    Returns:
        Tuple of (model, X_test, y_test)
    """
    if params is None:
        params = {
            "n_estimators": 200,
            "max_depth": 12,
            "random_state": random_state,
            "class_weight": "balanced",
            "n_jobs": -1,
        }

    # Chronological split
    target_col = "HOME_TEAM_WINS"
    date_col = "GAME_DATE_EST"

    features_sorted = features.copy()
    features_sorted[date_col] = pd.to_datetime(features_sorted[date_col], errors="coerce")
    features_sorted = features_sorted.sort_values(date_col).reset_index(drop=True)

    split_idx = int(len(features_sorted) * (1 - test_size))

    train_df = features_sorted.iloc[:split_idx].copy()
    test_df = features_sorted.iloc[split_idx:].copy()

    drop_cols = [target_col, date_col, "HOME_TEAM_ID", "VISITOR_TEAM_ID"]
    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[target_col].astype(int)

    X_test = test_df.drop(columns=drop_cols, errors="ignore")
    y_test = test_df[target_col].astype(int)

    use_mlflow = _configure_mlflow(mlflow_tracking_uri, mlflow_experiment_name)

    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    if use_mlflow:
        with mlflow.start_run(run_name="baseline_random_forest"):
            mlflow.log_params(params)
            mlflow.sklearn.log_model(model, name="model")
    else:
        warnings.warn(
            "Skipping baseline MLflow logging due to tracking backend issues.",
            UserWarning,
        )

    return model, X_test, y_test


def train_comparison_models(
    features: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    mlflow_tracking_uri: str = None,
    mlflow_experiment_name: str = None,
) -> Dict[str, Any]:
    """
    Train multiple models for comparison using GridSearchCV.
    
    Args:
        features: Feature dataset with target column
        test_size: Proportion of test data
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with trained models and their results
    """
    target_col = "HOME_TEAM_WINS"
    date_col = "GAME_DATE_EST"

    # Chronological split
    features_sorted = features.copy()
    features_sorted[date_col] = pd.to_datetime(features_sorted[date_col], errors="coerce")
    features_sorted = features_sorted.sort_values(date_col).reset_index(drop=True)

    split_idx = int(len(features_sorted) * (1 - test_size))

    train_df = features_sorted.iloc[:split_idx].copy()
    test_df = features_sorted.iloc[split_idx:].copy()

    drop_cols = [target_col, date_col, "HOME_TEAM_ID", "VISITOR_TEAM_ID"]
    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[target_col].astype(int)

    X_test = test_df.drop(columns=drop_cols, errors="ignore")
    y_test = test_df[target_col].astype(int)

    results = {}

    # Random Forest with GridSearchCV
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [8, 12],
        "min_samples_split": [2, 5],
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=random_state, class_weight="balanced", n_jobs=-1),
        rf_params,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)

    use_mlflow = _configure_mlflow(mlflow_tracking_uri, mlflow_experiment_name)

    if use_mlflow:
        with mlflow.start_run(run_name="rf_gridsearch"):
            mlflow.log_params(rf_grid.best_params_)
            mlflow.log_metric("best_cv_score", rf_grid.best_score_)
            mlflow.sklearn.log_model(rf_grid.best_estimator_, name="model")
    else:
        warnings.warn("Skipping RF MLflow logging due to tracking backend issues.", UserWarning)

    results["RandomForest"] = {
        "model": rf_grid.best_estimator_,
        "best_params": rf_grid.best_params_,
        "cv_score": rf_grid.best_score_,
    }

    # XGBoost with GridSearchCV
    xgb_params = {
        "n_estimators": [100, 200],
        "max_depth": [5, 7],
        "learning_rate": [0.01, 0.05],
    }
    xgb_grid = GridSearchCV(
        XGBClassifier(random_state=random_state, eval_metric="logloss"),
        xgb_params,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    xgb_grid.fit(X_train, y_train)

    if use_mlflow:
        with mlflow.start_run(run_name="xgb_gridsearch"):
            mlflow.log_params(xgb_grid.best_params_)
            mlflow.log_metric("best_cv_score", xgb_grid.best_score_)
            try:
                from mlflow import xgboost as mlflow_xgboost

                mlflow_xgboost.log_model(xgb_grid.best_estimator_, name="model")
            except Exception:
                mlflow.sklearn.log_model(
                    xgb_grid.best_estimator_,
                    name="model",
                    skops_trusted_types=[
                        "xgboost.core.Booster",
                        "xgboost.sklearn.XGBClassifier",
                    ],
                )
    else:
        warnings.warn("Skipping XGBoost MLflow logging due to tracking backend issues.", UserWarning)

    results["XGBoost"] = {
        "model": xgb_grid.best_estimator_,
        "best_params": xgb_grid.best_params_,
        "cv_score": xgb_grid.best_score_,
    }

    results["X_test"] = X_test
    results["y_test"] = y_test

    return results


def train_autogluon_model(
    features: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    autogluon_time_limit: int = 1800,
    autogluon_presets: str = "best_quality",
    mlflow_tracking_uri: str = None,
    mlflow_experiment_name: str = None,
) -> Tuple[Any, pd.DataFrame, pd.Series]:
    """
    Train an AutoGluon tabular model for NBA prediction.

    Returns:
        Tuple of (predictor, X_test, y_test)
    """
    target_col = "HOME_TEAM_WINS"
    date_col = "GAME_DATE_EST"

    features_sorted = features.copy()
    features_sorted[date_col] = pd.to_datetime(features_sorted[date_col], errors="coerce")
    features_sorted = features_sorted.sort_values(date_col).reset_index(drop=True)

    split_idx = int(len(features_sorted) * (1 - test_size))

    train_df = features_sorted.iloc[:split_idx].copy()
    test_df = features_sorted.iloc[split_idx:].copy()

    drop_cols = [target_col, date_col, "HOME_TEAM_ID", "VISITOR_TEAM_ID"]
    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[target_col].astype(int)

    X_test = test_df.drop(columns=drop_cols, errors="ignore")
    y_test = test_df[target_col].astype(int)

    # Create train_data with target column but without metadata columns
    # to match X_test structure for consistent predictions
    train_data = X_train.copy()
    train_data[target_col] = y_train
    train_data = TabularDataset(train_data)

    use_mlflow = _configure_mlflow(mlflow_tracking_uri, mlflow_experiment_name)

    predictor = TabularPredictor(
        label=target_col,
        path="models/autogluon_predictor",
        eval_metric="accuracy",
        verbosity=2,
    )
    predictor.fit(
        train_data,
        presets=autogluon_presets,
        num_stack_levels=2,
        time_limit=autogluon_time_limit,
    )

    if use_mlflow:
        with mlflow.start_run(run_name="autogluon_train"):
            mlflow.log_param("autogluon_presets", autogluon_presets)
            mlflow.log_param("autogluon_time_limit", autogluon_time_limit)
            mlflow.log_metric("train_shape", len(train_df))
            try:
                predictor.save("models/autogluon_predictor")
                mlflow.log_artifact("models/autogluon_predictor")
            except Exception:
                pass
    else:
        warnings.warn(
            "Skipping AutoGluon MLflow logging due to tracking backend issues.",
            UserWarning,
        )

    return predictor, X_test, y_test
