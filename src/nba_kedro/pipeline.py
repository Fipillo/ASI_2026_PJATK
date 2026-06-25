"""Kedro pipeline definition for NBA game prediction."""
from kedro.pipeline import Pipeline, node

from .nodes import (
    build_features,
    evaluate_model,
    load_details,
    load_games,
    train_baseline_model,
    train_comparison_models,
    train_autogluon_model,
)


def create_pipeline() -> Pipeline:
    """
    Create NBA prediction pipeline.
    
    Pipeline stages:
    1. Load raw games and details data
    2. Build pre-game features
    3. Train baseline and comparison models
    4. Evaluate models
    
    Returns:
        Kedro Pipeline object
    """
    return Pipeline(
        [
            # Data loading stage
            node(
                func=load_games,
                inputs="params:raw_data_path",
                outputs="games",
                name="load_games_node",
                tags=["data"],
            ),
            node(
                func=load_details,
                inputs="params:raw_data_path",
                outputs="details",
                name="load_details_node",
                tags=["data"],
            ),
            # Feature engineering stage
            node(
                func=build_features,
                inputs=["games", "details"],
                outputs="features",
                name="build_features_node",
                tags=["features"],
            ),
            # Model training stage
            node(
                func=train_baseline_model,
                inputs=[
                    "features",
                    "params:test_size",
                    "params:random_state",
                    "params:baseline_rf_params",
                    "params:mlflow_tracking_uri",
                    "params:mlflow_experiment_name",
                ],
                outputs=["baseline_model", "X_test_baseline", "y_test_baseline"],
                name="train_baseline_model_node",
                tags=["training"],
            ),
            node(
                func=train_comparison_models,
                inputs=[
                    "features",
                    "params:test_size",
                    "params:random_state",
                    "params:mlflow_tracking_uri",
                    "params:mlflow_experiment_name",
                ],
                outputs="comparison_models",
                name="train_comparison_models_node",
                tags=["training"],
            ),
            node(
                func=train_autogluon_model,
                inputs=[
                    "features",
                    "params:test_size",
                    "params:random_state",
                    "params:autogluon_time_limit",
                    "params:autogluon_presets",
                    "params:mlflow_tracking_uri",
                    "params:mlflow_experiment_name",
                ],
                outputs=["autogluon_model", "X_test_autogluon", "y_test_autogluon"],
                name="train_autogluon_model_node",
                tags=["training"],
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "autogluon_model",
                    "X_test_autogluon",
                    "y_test_autogluon",
                    "params:mlflow_tracking_uri",
                    "params:mlflow_experiment_name",
                ],
                outputs="autogluon_metrics",
                name="evaluate_autogluon_node",
                tags=["evaluation"],
            ),
            # Model evaluation stage
            node(
                func=evaluate_model,
                inputs=[
                    "baseline_model",
                    "X_test_baseline",
                    "y_test_baseline",
                    "params:mlflow_tracking_uri",
                    "params:mlflow_experiment_name",
                ],
                outputs="baseline_metrics",
                name="evaluate_baseline_node",
                tags=["evaluation"],
            ),
        ]
    )
