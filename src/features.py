from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


TARGET_COLUMN = "HOME_TEAM_WINS"


def load_games() -> pd.DataFrame:
    """Load raw NBA games dataset."""
    games_path = RAW_DIR / "games.csv"

    if not games_path.exists():
        raise FileNotFoundError(f"Missing file: {games_path}")

    games = pd.read_csv(games_path)
    return games


def prepare_basic_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a simple baseline feature dataset from games.csv.

    This is not the full feature engineering from the notebook.
    It is a lightweight local version used for the first Python pipeline.
    """

    df = games.copy()

    # Convert date and sort chronologically
    df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"], errors="coerce")
    df = df.sort_values("GAME_DATE_EST").reset_index(drop=True)

    # Keep only finished games with known target
    df = df[df["GAME_STATUS_TEXT"] == "Final"].copy()
    df = df.dropna(subset=[TARGET_COLUMN])

    # Basic numeric columns from games.csv
    numeric_columns = [
        "SEASON",
        "PTS_home",
        "FG_PCT_home",
        "FT_PCT_home",
        "FG3_PCT_home",
        "AST_home",
        "REB_home",
        "PTS_away",
        "FG_PCT_away",
        "FT_PCT_away",
        "FG3_PCT_away",
        "AST_away",
        "REB_away",
        TARGET_COLUMN,
    ]

    missing_columns = [col for col in numeric_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    features = df[numeric_columns].copy()

    # Simple differential features
    features["PTS_DIFF"] = features["PTS_home"] - features["PTS_away"]
    features["FG_PCT_DIFF"] = features["FG_PCT_home"] - features["FG_PCT_away"]
    features["FT_PCT_DIFF"] = features["FT_PCT_home"] - features["FT_PCT_away"]
    features["FG3_PCT_DIFF"] = features["FG3_PCT_home"] - features["FG3_PCT_away"]
    features["AST_DIFF"] = features["AST_home"] - features["AST_away"]
    features["REB_DIFF"] = features["REB_home"] - features["REB_away"]

    # Fill missing numeric values with median
    for col in features.columns:
        if col != TARGET_COLUMN:
            features[col] = features[col].fillna(features[col].median())

    features[TARGET_COLUMN] = features[TARGET_COLUMN].astype(int)

    return features


def save_features(features: pd.DataFrame, output_path: str = "data/processed/nba_features_basic.csv") -> None:
    """Save processed features to CSV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)


def main():
    print("Loading raw games data...")
    games = load_games()

    print(f"Raw games shape: {games.shape}")

    print("Preparing basic features...")
    features = prepare_basic_features(games)

    print(f"Prepared features shape: {features.shape}")
    print("Columns:")
    print(features.columns.tolist())

    output_path = "data/processed/nba_features_basic.csv"
    save_features(features, output_path)

    print(f"Saved features to: {output_path}")


if __name__ == "__main__":
    main()
