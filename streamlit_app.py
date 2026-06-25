from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/baseline_random_forest.pkl")
SAMPLE_PATH = Path("data/processed/nba_features_pregame_sample.csv")
TEAM_DETAILS_SAMPLE_PATH = Path("data/processed/games_details_sample.csv")
TEAM_DETAILS_PATH = Path("data/raw/games_details.csv")
TARGET_COLUMN = "HOME_TEAM_WINS"
DATE_COLUMN = "GAME_DATE_EST"

DEFAULT_TEAM_NAMES: dict[int, str] = {
    1610612737: "Boston Celtics (BOS)",
    1610612738: "Cleveland Cavaliers (CLE)",
    1610612739: "New York Knicks (NYK)",
    1610612740: "Philadelphia 76ers (PHI)",
    1610612741: "Chicago Bulls (CHI)",
    1610612742: "Dallas Mavericks (DAL)",
    1610612743: "Denver Nuggets (DEN)",
    1610612744: "Golden State Warriors (GSW)",
    1610612745: "Houston Rockets (HOU)",
    1610612746: "LA Clippers (LAC)",
    1610612747: "Los Angeles Lakers (LAL)",
    1610612748: "Miami Heat (MIA)",
    1610612749: "Milwaukee Bucks (MIL)",
    1610612750: "Minnesota Timberwolves (MIN)",
    1610612751: "New Orleans Pelicans (NOP)",
    1610612752: "Oklahoma City Thunder (OKC)",
    1610612753: "Orlando Magic (ORL)",
    1610612754: "Phoenix Suns (PHX)",
    1610612755: "Portland Trail Blazers (POR)",
    1610612756: "Sacramento Kings (SAC)",
    1610612757: "San Antonio Spurs (SAS)",
    1610612758: "Toronto Raptors (TOR)",
    1610612759: "Utah Jazz (UTA)",
    1610612760: "Washington Wizards (WAS)",
    1610612761: "Atlanta Hawks (ATL)",
    1610612762: "Brooklyn Nets (BKN)",
    1610612763: "Charlotte Hornets (CHA)",
    1610612764: "Detroit Pistons (DET)",
    1610612765: "Indiana Pacers (IND)",
    1610612766: "Memphis Grizzlies (MEM)",
}

TEAM_FEATURE_COLUMNS = [
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
    "ELO",
]

SEASON_OPTIONS = list(range(2003, 2027))

LABELS = {
    "REST_DAYS": "Dni odpoczynku",
    "LAST_GAME_WON": "Ostatni mecz wygrany",
    "SEASON_WIN_RATE": "Sezonowy wskaźnik wygranych",
    "ROLLING_PTS_5": "Śr. punktów (5)",
    "ROLLING_AST_5": "Śr. asyst (5)",
    "ROLLING_REB_5": "Śr. zbiórek (5)",
    "ROLLING_FG_PCT_5": "Śr. FG% (5)",
    "ROLLING_FT_PCT_5": "Śr. FT% (5)",
    "ROLLING_FG3_PCT_5": "Śr. FG3% (5)",
    "ROLLING_PTS_15": "Śr. punktów (15)",
    "ROLLING_AST_15": "Śr. asyst (15)",
    "ROLLING_REB_15": "Śr. zbiórek (15)",
    "ROLLING_FG_PCT_15": "Śr. FG% (15)",
    "ROLLING_FT_PCT_15": "Śr. FT% (15)",
    "ROLLING_FG3_PCT_15": "Śr. FG3% (15)",
    "ELO": "Elo pre-game",
}


def load_sample_data(path: Path = SAMPLE_PATH) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    return df


def load_team_metadata() -> dict[int, str]:
    metadata = DEFAULT_TEAM_NAMES.copy()
    path = TEAM_DETAILS_SAMPLE_PATH if TEAM_DETAILS_SAMPLE_PATH.exists() else TEAM_DETAILS_PATH
    if not path.exists():
        return metadata

    df = pd.read_csv(path, usecols=["TEAM_ID", "TEAM_CITY", "TEAM_ABBREVIATION"], low_memory=False)
    df = df.drop_duplicates(subset=["TEAM_ID"]).reset_index(drop=True)
    df["TEAM_NAME"] = df["TEAM_CITY"].fillna("") + " " + df["TEAM_ABBREVIATION"].fillna("")

    for team_id, team_name in zip(df["TEAM_ID"], df["TEAM_NAME"]):
        if pd.notna(team_id):
            team_id_int = int(team_id)
            if isinstance(team_name, str) and team_name.strip():
                if team_id_int not in metadata:
                    metadata[team_id_int] = team_name.strip()

    return metadata


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        return None
    return joblib.load(path)


def format_team_option(team_id: int, team_names: dict[int, str] | None) -> str:
    if team_names is None:
        return str(team_id)
    return team_names.get(team_id, str(team_id))


def get_default_team_inputs(sample_df: pd.DataFrame, prefix: str) -> dict[str, float]:
    defaults: dict[str, float] = {}
    if sample_df is None or sample_df.empty:
        for col in TEAM_FEATURE_COLUMNS:
            defaults[col] = 0.0
        return defaults

    row = sample_df.iloc[0]
    for col in TEAM_FEATURE_COLUMNS:
        defaults[col] = float(row.get(f"{prefix}_{col}", 0.0))
    return defaults


def build_prediction_row(home: dict[str, float], visitor: dict[str, float], season: int) -> pd.DataFrame:
    data: dict[str, float] = {}
    for col in TEAM_FEATURE_COLUMNS:
        data[f"HOME_{col}"] = home[col]
        data[f"VISITOR_{col}"] = visitor[col]

    data["SEASON"] = int(season)
    data["REST_DIFF"] = data["HOME_REST_DAYS"] - data["VISITOR_REST_DAYS"]
    data["WIN_RATE_DIFF"] = data["HOME_SEASON_WIN_RATE"] - data["VISITOR_SEASON_WIN_RATE"]
    data["ELO_DIFF"] = data["HOME_ELO"] - data["VISITOR_ELO"]

    diff_metrics = ["PTS", "AST", "REB", "FG_PCT", "FT_PCT", "FG3_PCT"]
    for metric in diff_metrics:
        data[f"{metric}_DIFF_5"] = data[f"HOME_ROLLING_{metric}_5"] - data[f"VISITOR_ROLLING_{metric}_5"]
        data[f"{metric}_DIFF_15"] = data[f"HOME_ROLLING_{metric}_15"] - data[f"VISITOR_ROLLING_{metric}_15"]

    return pd.DataFrame([data])


def render_team_inputs(title: str, defaults: dict[str, float], prefix: str) -> dict[str, float]:
    st.markdown(f"#### {title}")
    cols = st.columns(3)
    inputs: dict[str, float] = {}

    with cols[0]:
        inputs["REST_DAYS"] = st.number_input(
            "Dni odpoczynku",
            min_value=0.0,
            max_value=14.0,
            value=defaults["REST_DAYS"],
            step=1.0,
            format="%.0f",
            key=f"{prefix}_REST_DAYS",
        )
        inputs["LAST_GAME_WON"] = st.selectbox(
            "Ostatni mecz wygrany",
            [0.0, 0.5, 1.0],
            format_func=lambda x: "Nie" if x == 0.0 else ("Nieznany" if x == 0.5 else "Tak"),
            index={0.0: 0, 0.5: 1, 1.0: 2}.get(defaults["LAST_GAME_WON"], 1),
            key=f"{prefix}_LAST_GAME_WON",
        )
        inputs["SEASON_WIN_RATE"] = st.number_input(
            "Sezonowy wskaźnik wygranych",
            min_value=0.0,
            max_value=1.0,
            value=defaults["SEASON_WIN_RATE"],
            step=0.01,
            format="%.2f",
            key=f"{prefix}_SEASON_WIN_RATE",
        )
        inputs["ELO"] = st.number_input(
            "Elo pre-game",
            min_value=1000.0,
            max_value=3000.0,
            value=defaults["ELO"],
            step=1.0,
            format="%.0f",
            key=f"{prefix}_ELO",
        )

    with cols[1]:
        inputs["ROLLING_PTS_5"] = st.number_input(
            "Śr. punktów 5",
            min_value=0.0,
            value=defaults["ROLLING_PTS_5"],
            step=0.5,
            format="%.1f",
            key=f"{prefix}_ROLLING_PTS_5",
        )
        inputs["ROLLING_AST_5"] = st.number_input(
            "Śr. asyst 5",
            min_value=0.0,
            value=defaults["ROLLING_AST_5"],
            step=0.5,
            format="%.1f",
            key=f"{prefix}_ROLLING_AST_5",
        )
        inputs["ROLLING_REB_5"] = st.number_input(
            "Śr. zbiórek 5",
            min_value=0.0,
            value=defaults["ROLLING_REB_5"],
            step=0.5,
            format="%.1f",
            key=f"{prefix}_ROLLING_REB_5",
        )
        inputs["ROLLING_FG_PCT_5"] = st.number_input(
            "Śr. FG% 5",
            min_value=0.0,
            max_value=1.0,
            value=defaults["ROLLING_FG_PCT_5"],
            step=0.01,
            format="%.2f",
            key=f"{prefix}_ROLLING_FG_PCT_5",
        )
        inputs["ROLLING_FT_PCT_5"] = st.number_input(
            "Śr. FT% 5",
            min_value=0.0,
            max_value=1.0,
            value=defaults["ROLLING_FT_PCT_5"],
            step=0.01,
            format="%.2f",
            key=f"{prefix}_ROLLING_FT_PCT_5",
        )
        inputs["ROLLING_FG3_PCT_5"] = st.number_input(
            "Śr. FG3% 5",
            min_value=0.0,
            max_value=1.0,
            value=defaults["ROLLING_FG3_PCT_5"],
            step=0.01,
            format="%.2f",
            key=f"{prefix}_ROLLING_FG3_PCT_5",
        )

    with cols[2]:
        inputs["ROLLING_PTS_15"] = st.number_input(
            "Śr. punktów 15",
            min_value=0.0,
            value=defaults["ROLLING_PTS_15"],
            step=0.5,
            format="%.1f",
            key=f"{prefix}_ROLLING_PTS_15",
        )
        inputs["ROLLING_AST_15"] = st.number_input(
            "Śr. asyst 15",
            min_value=0.0,
            value=defaults["ROLLING_AST_15"],
            step=0.5,
            format="%.1f",
            key=f"{prefix}_ROLLING_AST_15",
        )
        inputs["ROLLING_REB_15"] = st.number_input(
            "Śr. zbiórek 15",
            min_value=0.0,
            value=defaults["ROLLING_REB_15"],
            step=0.5,
            format="%.1f",
            key=f"{prefix}_ROLLING_REB_15",
        )
        inputs["ROLLING_FG_PCT_15"] = st.number_input(
            "Śr. FG% 15",
            min_value=0.0,
            max_value=1.0,
            value=defaults["ROLLING_FG_PCT_15"],
            step=0.01,
            format="%.2f",
            key=f"{prefix}_ROLLING_FG_PCT_15",
        )
        inputs["ROLLING_FT_PCT_15"] = st.number_input(
            "Śr. FT% 15",
            min_value=0.0,
            max_value=1.0,
            value=defaults["ROLLING_FT_PCT_15"],
            step=0.01,
            format="%.2f",
            key=f"{prefix}_ROLLING_FT_PCT_15",
        )
        inputs["ROLLING_FG3_PCT_15"] = st.number_input(
            "Śr. FG3% 15",
            min_value=0.0,
            max_value=1.0,
            value=defaults["ROLLING_FG3_PCT_15"],
            step=0.01,
            format="%.2f",
            key=f"{prefix}_ROLLING_FG3_PCT_15",
        )

    return inputs


def main() -> None:
    st.set_page_config(
        page_title="NBA Game Winner Prediction",
        page_icon="🏀",
        layout="wide",
    )

    st.title("NBA Game Winner Prediction")
    st.write(
        "Interaktywny prototyp Streamlit do oceny predykcji wyniku meczu NBA. "
        "Aplikacja przyjmuje dane dwóch drużyn i zwraca prognozę zwycięzcy. "
    )

    sample_data = load_sample_data()
    team_names = load_team_metadata()
    model = load_model()

    if sample_data is None:
        st.error(
            "Nie znaleziono pliku `data/processed/nba_features_pregame_sample.csv`. "
            "Przygotuj dane lub uruchom pipeline w katalogu `data/processed`."
        )
        return

    team_ids = sorted(team_names.keys(), key=lambda tid: format_team_option(tid, team_names))
    has_team_selection = len(team_ids) >= 2

    home_defaults = get_default_team_inputs(sample_data, "HOME")
    visitor_defaults = get_default_team_inputs(sample_data, "VISITOR")

    with st.form("prediction_form"):
        st.subheader("Wypełnij dane meczu")

        current_season = sample_data["SEASON"].mode().iloc[0] if "SEASON" in sample_data.columns else 2003
        season = st.selectbox(
            "Sezon",
            SEASON_OPTIONS,
            index=SEASON_OPTIONS.index(int(current_season)) if int(current_season) in SEASON_OPTIONS else len(SEASON_OPTIONS) - 1,
            help="Wybierz sezon dla którego chcesz wykonać prognozę.",
            key="season",
        )

        if has_team_selection:
            home_team_id = st.selectbox(
                "Drużyna gospodarzy",
                team_ids,
                format_func=lambda tid: format_team_option(tid, team_names),
            )
            visitor_team_ids = [tid for tid in team_ids if tid != home_team_id]
            if not visitor_team_ids:
                visitor_team_ids = team_ids
            visitor_team_id = st.selectbox(
                "Drużyna gości",
                visitor_team_ids,
                format_func=lambda tid: format_team_option(tid, team_names),
            )
        else:
            home_team_id = None
            visitor_team_id = None
            st.info(
                "Brak identyfikatorów drużyn w metadanych. Wprowadź wartości cech ręcznie."
            )

        st.markdown("---")
        home_inputs = render_team_inputs("Dane gospodarzy", home_defaults, "home")
        st.markdown("---")
        visitor_inputs = render_team_inputs("Dane gości", visitor_defaults, "visitor")

        submitted = st.form_submit_button("Oblicz prognozę")

    if not submitted:
        st.info("Wypełnij formularz i naciśnij przycisk, aby otrzymać wynik predykcji.")
        return

    if model is None:
        st.error("Nie znaleziono modelu. Umieść plik `models/baseline_random_forest.pkl` lub wytrenuj model.")
        return

    season = st.session_state.get("season", sample_data["SEASON"].mode().iloc[0] if "SEASON" in sample_data.columns else 2003)
    features = build_prediction_row(home_inputs, visitor_inputs, season)

    model_feature_order = [
        col for col in sample_data.columns
        if col not in {"GAME_DATE_EST", "HOME_TEAM_WINS", "HOME_TEAM_ID", "VISITOR_TEAM_ID"}
    ]
    features = features[model_feature_order]

    proba = model.predict_proba(features)[0]
    home_win_probability = float(proba[1])
    away_win_probability = float(proba[0])
    predicted_winner = "Gospodarze" if home_win_probability >= away_win_probability else "Goście"

    st.markdown("### Wynik predykcji")
    st.metric("Przewidywany zwycięzca", predicted_winner)
    st.metric("Prawdopodobieństwo gospodarzy", f"{home_win_probability:.1%}")
    st.metric("Prawdopodobieństwo gości", f"{away_win_probability:.1%}")

    st.markdown("### Wybrane dane")
    st.write(
        f"- Gospodarze: **{format_team_option(home_team_id, team_names) if home_team_id is not None else 'Brak'}**  \n"
        f"- Goście: **{format_team_option(visitor_team_id, team_names) if visitor_team_id is not None else 'Brak'}**"
    )

    st.markdown("---")
    st.subheader("Cechy użyte do prognozy")
    st.write("W tabeli poniżej znajdują się wartości wejściowe dla gospodarzy i gości oraz policzone różnice.")
    st.dataframe(features, use_container_width=True)


if __name__ == "__main__":
    main()
