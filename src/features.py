from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

TARGET_COLUMN = "HOME_TEAM_WINS"


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw NBA games and game details datasets."""
    games_path = RAW_DIR / "games.csv"
    details_path = RAW_DIR / "games_details.csv"

    if not games_path.exists():
        raise FileNotFoundError(f"Missing file: {games_path}")

    if not details_path.exists():
        raise FileNotFoundError(f"Missing file: {details_path}")

    games = pd.read_csv(games_path)
    details = pd.read_csv(details_path, low_memory=False)

    games["GAME_DATE_EST"] = pd.to_datetime(games["GAME_DATE_EST"], errors="coerce")
    games = games.sort_values("GAME_DATE_EST").reset_index(drop=True)

    return games, details


def build_team_stats(details: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-level game details into team-level game statistics."""
    team_stats = (
        details.groupby(["GAME_ID", "TEAM_ID"])
        .agg(
            PTS=("PTS", "sum"),
            AST=("AST", "sum"),
            REB=("REB", "sum"),
            FG_PCT=("FG_PCT", "mean"),
            FT_PCT=("FT_PCT", "mean"),
            FG3_PCT=("FG3_PCT", "mean"),
        )
        .reset_index()
    )

    return team_stats


def build_team_performance(games: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Build long-form team performance table.

    Each game becomes two rows:
    - one for home team,
    - one for visitor team.
    """
    rows = []

    for _, row in games.iterrows():
        rows.append(
            {
                "GAME_ID": row["GAME_ID"],
                "TEAM_ID": row["HOME_TEAM_ID"],
                "OPPONENT_ID": row["VISITOR_TEAM_ID"],
                "GAME_DATE_EST": row["GAME_DATE_EST"],
                "SEASON": row["SEASON"],
                "WON": row[TARGET_COLUMN],
                "IS_HOME": 1,
            }
        )

        rows.append(
            {
                "GAME_ID": row["GAME_ID"],
                "TEAM_ID": row["VISITOR_TEAM_ID"],
                "OPPONENT_ID": row["HOME_TEAM_ID"],
                "GAME_DATE_EST": row["GAME_DATE_EST"],
                "SEASON": row["SEASON"],
                "WON": 1 - row[TARGET_COLUMN],
                "IS_HOME": 0,
            }
        )

    performance = pd.DataFrame(rows)
    performance = performance.merge(team_stats, on=["GAME_ID", "TEAM_ID"], how="left")
    performance = performance.sort_values(["TEAM_ID", "GAME_DATE_EST"]).reset_index(drop=True)

    return performance


def add_team_history_features(performance: pd.DataFrame) -> pd.DataFrame:
    """
    Add pre-game historical features.

    All rolling and previous-game features are shifted by one game,
    so they use only information available before the current match.
    """
    df = performance.copy()
    df = df.sort_values(["TEAM_ID", "GAME_DATE_EST"]).reset_index(drop=True)

    df["REST_DAYS"] = (
        df.groupby("TEAM_ID")["GAME_DATE_EST"]
        .diff()
        .dt.days
        .fillna(7)
        .clip(lower=0, upper=7)
    )

    df["LAST_GAME_WON"] = (
        df.groupby("TEAM_ID")["WON"]
        .shift(1)
        .fillna(0.5)
    )

    rolling_columns = ["PTS", "AST", "REB", "FG_PCT", "FT_PCT", "FG3_PCT"]

    for window in [5, 15]:
        for col in rolling_columns:
            df[f"ROLLING_{col}_{window}"] = (
                df.groupby("TEAM_ID")[col]
                .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
            )

    df["SEASON_GAMES_BEFORE"] = df.groupby(["TEAM_ID", "SEASON"]).cumcount()

    df["SEASON_WINS_BEFORE"] = (
        df.groupby(["TEAM_ID", "SEASON"])["WON"]
        .transform(lambda s: s.shift(1).cumsum())
        .fillna(0)
    )

    df["SEASON_WIN_RATE"] = (
        df["SEASON_WINS_BEFORE"] / df["SEASON_GAMES_BEFORE"].replace(0, np.nan)
    ).fillna(0.5)

    for col in df.columns:
        if col.startswith("ROLLING_"):
            df[col] = df[col].fillna(df[col].median())

    return df


def calculate_elo(games: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """
    Calculate pre-game Elo rating.

    Elo is recorded before updating ratings after the game,
    so it does not use the current game result as an input feature.
    """
    all_teams = pd.concat([games["HOME_TEAM_ID"], games["VISITOR_TEAM_ID"]]).unique()
    elo_ratings = {team_id: 1500.0 for team_id in all_teams}
    elo_history = []

    for _, row in games.sort_values("GAME_DATE_EST").iterrows():
        home_id = row["HOME_TEAM_ID"]
        visitor_id = row["VISITOR_TEAM_ID"]

        home_elo = elo_ratings[home_id]
        visitor_elo = elo_ratings[visitor_id]

        elo_history.append(
            {
                "GAME_ID": row["GAME_ID"],
                "HOME_ELO": home_elo,
                "VISITOR_ELO": visitor_elo,
            }
        )

        expected_home = 1 / (1 + 10 ** ((visitor_elo - home_elo) / 400))
        actual_home = row[TARGET_COLUMN]

        elo_ratings[home_id] += k * (actual_home - expected_home)
        elo_ratings[visitor_id] += k * ((1 - actual_home) - (1 - expected_home))

    return pd.DataFrame(elo_history)


def build_match_features(
    games: pd.DataFrame,
    performance: pd.DataFrame,
    elo: pd.DataFrame,
) -> pd.DataFrame:
    """Merge home and visitor team features into one row per game."""
    team_feature_columns = [
        "GAME_ID",
        "TEAM_ID",
        "REST_DAYS",
        "LAST_GAME_WON",
        "SEASON_WIN_RATE",
        "ROLLING_PTS_5",
        "ROLLING_AST_5",
        "ROLLING_REB_5",
        "ROLLING_FG_PCT_5",
        "ROLLING_FT_PCT_5",
        "ROLLING_FG3_PCT_5",
        "ROLLING_PTS_15",
        "ROLLING_AST_15",
        "ROLLING_REB_15",
        "ROLLING_FG_PCT_15",
        "ROLLING_FT_PCT_15",
        "ROLLING_FG3_PCT_15",
    ]

    team_features = performance[team_feature_columns].copy()

    base = games[
        [
            "GAME_ID",
            "GAME_DATE_EST",
            "SEASON",
            "HOME_TEAM_ID",
            "VISITOR_TEAM_ID",
            TARGET_COLUMN,
        ]
    ].copy()

    output = base.merge(
        team_features,
        left_on=["GAME_ID", "HOME_TEAM_ID"],
        right_on=["GAME_ID", "TEAM_ID"],
        how="left",
    )

    home_rename = {
        col: f"HOME_{col}"
        for col in team_feature_columns
        if col not in ["GAME_ID", "TEAM_ID"]
    }

    output = output.rename(columns=home_rename).drop(columns=["TEAM_ID"])

    output = output.merge(
        team_features,
        left_on=["GAME_ID", "VISITOR_TEAM_ID"],
        right_on=["GAME_ID", "TEAM_ID"],
        how="left",
    )

    visitor_rename = {
        col: f"VISITOR_{col}"
        for col in team_feature_columns
        if col not in ["GAME_ID", "TEAM_ID"]
    }

    output = output.rename(columns=visitor_rename).drop(columns=["TEAM_ID"])
    output = output.merge(elo, on="GAME_ID", how="left")

    output["REST_DIFF"] = output["HOME_REST_DAYS"] - output["VISITOR_REST_DAYS"]
    output["WIN_RATE_DIFF"] = output["HOME_SEASON_WIN_RATE"] - output["VISITOR_SEASON_WIN_RATE"]
    output["ELO_DIFF"] = output["HOME_ELO"] - output["VISITOR_ELO"]

    for col in ["PTS", "AST", "REB", "FG_PCT", "FT_PCT", "FG3_PCT"]:
        output[f"{col}_DIFF_5"] = (
            output[f"HOME_ROLLING_{col}_5"] - output[f"VISITOR_ROLLING_{col}_5"]
        )
        output[f"{col}_DIFF_15"] = (
            output[f"HOME_ROLLING_{col}_15"] - output[f"VISITOR_ROLLING_{col}_15"]
        )

    output = output.drop(columns=["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID"])

    for col in output.columns:
        if col not in ["GAME_DATE_EST", TARGET_COLUMN]:
            output[col] = output[col].fillna(output[col].median())

    output[TARGET_COLUMN] = output[TARGET_COLUMN].astype(int)

    return output


def save_outputs(features: pd.DataFrame) -> None:
    """Save full feature dataset, sample dataset and column list."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    full_path = PROCESSED_DIR / "nba_features_pregame.csv"
    sample_path = PROCESSED_DIR / "nba_features_pregame_sample.csv"
    columns_path = PROCESSED_DIR / "nba_features_pregame_columns.txt"

    features.to_csv(full_path, index=False)
    features.head(100).to_csv(sample_path, index=False)

    with open(columns_path, "w") as f:
        for col in features.columns:
            f.write(col + "\n")

    print(f"Saved full dataset: {full_path}")
    print(f"Saved sample dataset: {sample_path}")
    print(f"Saved columns list: {columns_path}")


def main() -> None:
    print("Loading raw data...")
    games, details = load_raw_data()

    print(f"games shape: {games.shape}")
    print(f"details shape: {details.shape}")

    print("Building team stats...")
    team_stats = build_team_stats(details)
    print(f"team stats shape: {team_stats.shape}")

    print("Building team performance...")
    performance = build_team_performance(games, team_stats)
    print(f"team performance shape: {performance.shape}")

    print("Adding historical pre-game features...")
    performance = add_team_history_features(performance)

    print("Calculating Elo ratings...")
    elo = calculate_elo(games)
    print(f"elo shape: {elo.shape}")

    print("Building match-level feature dataset...")
    features = build_match_features(games, performance, elo)

    print(f"final features shape: {features.shape}")
    print("columns:")
    print(features.columns.tolist())

    save_outputs(features)

    print("Feature engineering finished successfully.")


if __name__ == "__main__":
    main()
