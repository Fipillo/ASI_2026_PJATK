"""Baseline model training module."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


FEATURES_PATH = Path("data/processed/nba_features_pregame.csv")
MODEL_PATH = Path("models/baseline_random_forest.pkl")
METRICS_PATH = Path("reports/baseline_metrics.csv")

TARGET_COLUMN = "HOME_TEAM_WINS"
DATE_COLUMN = "GAME_DATE_EST"


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Load processed pre-game feature dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed features file: {path}. "
            "Run `python -m src.features` first."
        )

    df = pd.read_csv(path)
    return df


def chronological_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split data chronologically.

    For sports data this is better than random split, because the model should train
    on older games and be evaluated on newer games.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    if DATE_COLUMN not in df.columns:
        raise ValueError(f"Date column '{DATE_COLUMN}' not found.")

    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    x_train = train_df.drop(
        columns=[TARGET_COLUMN, DATE_COLUMN, "HOME_TEAM_ID", "VISITOR_TEAM_ID"],
        errors="ignore",
    )
    y_train = train_df[TARGET_COLUMN].astype(int)

    x_test = test_df.drop(
        columns=[TARGET_COLUMN, DATE_COLUMN, "HOME_TEAM_ID", "VISITOR_TEAM_ID"],
        errors="ignore",
    )
    y_test = test_df[TARGET_COLUMN].astype(int)

    return x_train, x_test, y_train, y_test


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train a simple baseline Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate trained model on test data."""
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    metrics = {
        "model": "RandomForestClassifier",
        "split": "chronological_80_20",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "test_rows": len(x_test),
        "features_count": x_test.shape[1],
    }

    return metrics


def save_model(model, path: Path = MODEL_PATH) -> None:
    """Save trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_metrics(metrics: dict, path: Path = METRICS_PATH) -> None:
    """Save metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(path, index=False)


def main() -> None:
    print("Loading pre-game features...")
    df = load_features()
    print(f"Dataset shape: {df.shape}")

    print("Splitting data chronologically...")
    x_train, x_test, y_train, y_test = chronological_train_test_split(df)
    print(f"Train shape: {x_train.shape}")
    print(f"Test shape: {x_test.shape}")

    print("Training baseline Random Forest model...")
    model = train_model(x_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_model(model, x_test, y_test)

    print("Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("Saving model...")
    save_model(model)

    print("Saving metrics...")
    save_metrics(metrics)

    print("Training finished successfully.")


if __name__ == "__main__":
    main()
