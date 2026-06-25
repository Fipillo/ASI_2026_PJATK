"""Unit tests for src/api/main.py."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(name="mock_model")
def fixture_mock_model():
    """Sklearn-compatible mock that always predicts home team wins (class 1)."""
    model = MagicMock()
    model.predict.return_value = [1]
    model.predict_proba.return_value = [[0.30, 0.70]]
    return model


@pytest.fixture(name="client")
def fixture_client(mock_model):
    """FastAPI TestClient with the model loader mocked out."""

    def fake_load():
        import src.model_loader as loader  # pylint: disable=import-outside-toplevel

        loader._model = mock_model

    with patch("src.model_loader.load", side_effect=fake_load):
        from src.api.main import app  # pylint: disable=import-outside-toplevel

        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture(name="valid_payload")
def fixture_valid_payload():
    """Complete valid request payload covering all 48 model features."""
    return {
        "game_date": "2023-01-15",
        "home_team": "LAL",
        "visitor_team": "BOS",
        "season": 2023,
        # Home features
        "home_rest_days": 2.0,
        "home_last_game_won": 1.0,
        "home_season_win_rate": 0.60,
        "home_rolling_pts_5": 112.0,
        "home_rolling_ast_5": 25.0,
        "home_rolling_reb_5": 44.0,
        "home_rolling_fg_pct_5": 0.48,
        "home_rolling_ft_pct_5": 0.78,
        "home_rolling_fg3_pct_5": 0.36,
        "home_rolling_pts_15": 110.0,
        "home_rolling_ast_15": 24.0,
        "home_rolling_reb_15": 43.0,
        "home_rolling_fg_pct_15": 0.47,
        "home_rolling_ft_pct_15": 0.77,
        "home_rolling_fg3_pct_15": 0.35,
        # Visitor features
        "visitor_rest_days": 1.0,
        "visitor_last_game_won": 0.0,
        "visitor_season_win_rate": 0.40,
        "visitor_rolling_pts_5": 105.0,
        "visitor_rolling_ast_5": 22.0,
        "visitor_rolling_reb_5": 41.0,
        "visitor_rolling_fg_pct_5": 0.45,
        "visitor_rolling_ft_pct_5": 0.75,
        "visitor_rolling_fg3_pct_5": 0.33,
        "visitor_rolling_pts_15": 104.0,
        "visitor_rolling_ast_15": 21.0,
        "visitor_rolling_reb_15": 40.0,
        "visitor_rolling_fg_pct_15": 0.44,
        "visitor_rolling_ft_pct_15": 0.74,
        "visitor_rolling_fg3_pct_15": 0.32,
        # Elo
        "home_elo": 1520.0,
        "visitor_elo": 1480.0,
        # Differential features
        "rest_diff": 1.0,
        "win_rate_diff": 0.20,
        "elo_diff": 40.0,
        "pts_diff_5": 7.0,
        "pts_diff_15": 6.0,
        "ast_diff_5": 3.0,
        "ast_diff_15": 3.0,
        "reb_diff_5": 3.0,
        "reb_diff_15": 3.0,
        "fg_pct_diff_5": 0.03,
        "fg_pct_diff_15": 0.03,
        "ft_pct_diff_5": 0.03,
        "ft_pct_diff_15": 0.03,
        "fg3_pct_diff_5": 0.03,
        "fg3_pct_diff_15": 0.03,
    }


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------


def test_predict_returns_200(client, valid_payload):
    """A well-formed request must return HTTP 200."""
    response = client.post("/api/v1/predict", json=valid_payload)
    assert response.status_code == 200


def test_predict_response_fields_present(client, valid_payload):
    """Response body must contain all required fields."""
    response = client.post("/api/v1/predict", json=valid_payload)
    data = response.json()
    for field in ("prediction", "confidence", "probability", "game_id", "processed_at"):
        assert field in data, f"Missing field: {field}"


def test_predict_home_team_wins_label(client, valid_payload):
    """When the model predicts class 1, label must be 'home_team_wins'."""
    response = client.post("/api/v1/predict", json=valid_payload)
    assert response.json()["prediction"] == "home_team_wins"


def test_predict_confidence_in_range(client, valid_payload):
    """Confidence score must be in [0.0, 1.0]."""
    response = client.post("/api/v1/predict", json=valid_payload)
    confidence = response.json()["confidence"]
    assert 0.0 <= confidence <= 1.0


def test_predict_probability_both_keys(client, valid_payload):
    """Probability dict must include entries for both teams."""
    response = client.post("/api/v1/predict", json=valid_payload)
    prob = response.json()["probability"]
    assert "home_team_wins" in prob
    assert "visitor_team_wins" in prob


def test_predict_game_id_contains_teams(client, valid_payload):
    """game_id must embed both team codes."""
    response = client.post("/api/v1/predict", json=valid_payload)
    game_id = response.json()["game_id"]
    assert "LAL" in game_id
    assert "BOS" in game_id


def test_predict_team_code_normalised_to_uppercase(client, valid_payload):
    """Lowercase team codes in the request must be uppercased in game_id."""
    valid_payload["home_team"] = "lal"
    valid_payload["visitor_team"] = "bos"
    response = client.post("/api/v1/predict", json=valid_payload)
    assert response.status_code == 200
    assert "LAL" in response.json()["game_id"]


# ---------------------------------------------------------------------------
# Validation errors (HTTP 422)
# ---------------------------------------------------------------------------


def test_predict_rejects_fg_pct_above_one(client, valid_payload):
    """FG percentage above 1.0 must fail Pydantic validation."""
    valid_payload["home_rolling_fg_pct_5"] = 1.5
    response = client.post("/api/v1/predict", json=valid_payload)
    assert response.status_code == 422


def test_predict_rejects_rest_days_above_seven(client, valid_payload):
    """Rest days above 7.0 must fail Pydantic validation."""
    valid_payload["home_rest_days"] = 10.0
    response = client.post("/api/v1/predict", json=valid_payload)
    assert response.status_code == 422


def test_predict_rejects_negative_pts(client, valid_payload):
    """Negative rolling points must fail Pydantic validation."""
    valid_payload["home_rolling_pts_5"] = -5.0
    response = client.post("/api/v1/predict", json=valid_payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Error propagation (HTTP 500)
# ---------------------------------------------------------------------------


def test_predict_returns_500_when_inference_fails(client, valid_payload):
    """A RuntimeError from the model must surface as HTTP 500."""
    with patch("src.model_loader.predict", side_effect=RuntimeError("Model not loaded")):
        response = client.post("/api/v1/predict", json=valid_payload)
        assert response.status_code == 500
