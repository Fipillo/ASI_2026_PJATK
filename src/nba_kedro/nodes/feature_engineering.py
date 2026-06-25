"""Feature engineering nodes for NBA pipeline."""
import numpy as np
import pandas as pd


def build_features(
    games: pd.DataFrame,
    details: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build pre-game features from raw games and details data.
    
    Implements:
    - Team statistics aggregation
    - Rest days calculation
    - Rolling averages (5-game and 15-game)
    - Season win rates
    - Elo ratings
    - Feature differences (home vs visitor)
    
    Args:
        games: Games dataset
        details: Game details dataset
        
    Returns:
        DataFrame with match-level pre-game features
    """
    # Team stats aggregation
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

    # Build team performance table (long-form)
    rows = []
    for _, row in games.iterrows():
        rows.append(
            {
                "GAME_ID": row["GAME_ID"],
                "TEAM_ID": row["HOME_TEAM_ID"],
                "OPPONENT_ID": row["VISITOR_TEAM_ID"],
                "GAME_DATE_EST": row["GAME_DATE_EST"],
                "SEASON": row["SEASON"],
                "WON": row["HOME_TEAM_WINS"],
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
                "WON": 1 - row["HOME_TEAM_WINS"],
                "IS_HOME": 0,
            }
        )

    performance = pd.DataFrame(rows)
    performance = performance.merge(team_stats, on=["GAME_ID", "TEAM_ID"], how="left")

    # Ensure dates are datetime before calculating rest days
    performance["GAME_DATE_EST"] = pd.to_datetime(performance["GAME_DATE_EST"], errors="coerce")
    performance = performance.sort_values(["TEAM_ID", "GAME_DATE_EST"]).reset_index(drop=True)

    # Rest days
    performance["REST_DAYS"] = (
        performance.groupby("TEAM_ID")["GAME_DATE_EST"]
        .diff()
        .dt.days
        .fillna(7)
        .clip(lower=0, upper=7)
    )

    # Last game won
    performance["LAST_GAME_WON"] = (
        performance.groupby("TEAM_ID")["WON"]
        .shift(1)
        .fillna(0.5)
    )

    # Rolling averages
    rolling_columns = ["PTS", "AST", "REB", "FG_PCT", "FT_PCT", "FG3_PCT"]
    for window in [5, 15]:
        for col in rolling_columns:
            performance[f"ROLLING_{col}_{window}"] = (
                performance.groupby("TEAM_ID")[col]
                .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
            )

    # Season win rate
    performance["SEASON_GAMES_BEFORE"] = performance.groupby(["TEAM_ID", "SEASON"]).cumcount()
    performance["SEASON_WINS_BEFORE"] = (
        performance.groupby(["TEAM_ID", "SEASON"])["WON"]
        .transform(lambda s: s.shift(1).cumsum())
        .fillna(0)
    )
    performance["SEASON_WIN_RATE"] = (
        performance["SEASON_WINS_BEFORE"] / performance["SEASON_GAMES_BEFORE"].replace(0, np.nan)
    ).fillna(0.5)

    # Fill NaN with median
    for col in performance.columns:
        if col.startswith("ROLLING_"):
            performance[col] = performance[col].fillna(performance[col].median())

    # Calculate Elo ratings
    elo_df = _calculate_elo(games, k=20)

    # Build match-level features
    features = games[
        ["GAME_ID", "GAME_DATE_EST", "SEASON", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "HOME_TEAM_WINS"]
    ].copy()

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

    # Merge home features
    features = features.merge(
        team_features,
        left_on=["GAME_ID", "HOME_TEAM_ID"],
        right_on=["GAME_ID", "TEAM_ID"],
        how="left",
    )
    features = features.rename(
        columns={col: f"HOME_{col}" for col in team_feature_columns if col not in ["GAME_ID", "TEAM_ID"]}
    ).drop(columns=["TEAM_ID"])

    # Merge visitor features
    features = features.merge(
        team_features,
        left_on=["GAME_ID", "VISITOR_TEAM_ID"],
        right_on=["GAME_ID", "TEAM_ID"],
        how="left",
    )
    features = features.rename(
        columns={col: f"VISITOR_{col}" for col in team_feature_columns if col not in ["GAME_ID", "TEAM_ID"]}
    ).drop(columns=["TEAM_ID"])

    # Merge Elo ratings
    features = features.merge(elo_df, on="GAME_ID", how="left")

    # Calculate differences
    features["REST_DIFF"] = features["HOME_REST_DAYS"] - features["VISITOR_REST_DAYS"]
    features["WIN_RATE_DIFF"] = features["HOME_SEASON_WIN_RATE"] - features["VISITOR_SEASON_WIN_RATE"]
    features["ELO_DIFF"] = features["HOME_ELO"] - features["VISITOR_ELO"]

    for col in ["PTS", "AST", "REB", "FG_PCT", "FT_PCT", "FG3_PCT"]:
        features[f"{col}_DIFF_5"] = features[f"HOME_ROLLING_{col}_5"] - features[f"VISITOR_ROLLING_{col}_5"]
        features[f"{col}_DIFF_15"] = features[f"HOME_ROLLING_{col}_15"] - features[f"VISITOR_ROLLING_{col}_15"]

    # Keep team identifiers for downstream matching, but not for training.
    features = features.drop(columns=["GAME_ID"])

    # Fill remaining NaN
    for col in features.columns:
        if col not in ["GAME_DATE_EST", "HOME_TEAM_WINS"]:
            features[col] = features[col].fillna(features[col].median())

    features["HOME_TEAM_WINS"] = features["HOME_TEAM_WINS"].astype(int)

    return features


def _calculate_elo(games: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Calculate Elo ratings before each game."""
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
        actual_home = row["HOME_TEAM_WINS"]

        elo_ratings[home_id] += k * (actual_home - expected_home)
        elo_ratings[visitor_id] += k * ((1 - actual_home) - (1 - expected_home))

    return pd.DataFrame(elo_history)
