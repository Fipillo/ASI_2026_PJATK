from datetime import date

import numpy as np
from pydantic import BaseModel, Field, field_validator

from src.api.config import FEATURE_COLUMNS


class GamePredictionRequest(BaseModel):
    """Request model for NBA game prediction endpoint."""

    game_date: date
    home_team: str
    visitor_team: str

    season: int = Field(ge=2000)

    home_rest_days: float = Field(ge=0.0, le=7.0)
    home_last_game_won: float = Field(ge=0.0, le=1.0)
    home_season_win_rate: float = Field(ge=0.0, le=1.0)
    home_rolling_pts_5: float = Field(ge=0.0)
    home_rolling_ast_5: float = Field(ge=0.0)
    home_rolling_reb_5: float = Field(ge=0.0)
    home_rolling_fg_pct_5: float = Field(ge=0.0, le=1.0)
    home_rolling_ft_pct_5: float = Field(ge=0.0, le=1.0)
    home_rolling_fg3_pct_5: float = Field(ge=0.0, le=1.0)
    home_rolling_pts_15: float = Field(ge=0.0)
    home_rolling_ast_15: float = Field(ge=0.0)
    home_rolling_reb_15: float = Field(ge=0.0)
    home_rolling_fg_pct_15: float = Field(ge=0.0, le=1.0)
    home_rolling_ft_pct_15: float = Field(ge=0.0, le=1.0)
    home_rolling_fg3_pct_15: float = Field(ge=0.0, le=1.0)

    visitor_rest_days: float = Field(ge=0.0, le=7.0)
    visitor_last_game_won: float = Field(ge=0.0, le=1.0)
    visitor_season_win_rate: float = Field(ge=0.0, le=1.0)
    visitor_rolling_pts_5: float = Field(ge=0.0)
    visitor_rolling_ast_5: float = Field(ge=0.0)
    visitor_rolling_reb_5: float = Field(ge=0.0)
    visitor_rolling_fg_pct_5: float = Field(ge=0.0, le=1.0)
    visitor_rolling_ft_pct_5: float = Field(ge=0.0, le=1.0)
    visitor_rolling_fg3_pct_5: float = Field(ge=0.0, le=1.0)
    visitor_rolling_pts_15: float = Field(ge=0.0)
    visitor_rolling_ast_15: float = Field(ge=0.0)
    visitor_rolling_reb_15: float = Field(ge=0.0)
    visitor_rolling_fg_pct_15: float = Field(ge=0.0, le=1.0)
    visitor_rolling_ft_pct_15: float = Field(ge=0.0, le=1.0)
    visitor_rolling_fg3_pct_15: float = Field(ge=0.0, le=1.0)

    home_elo: float = Field(ge=0.0)
    visitor_elo: float = Field(ge=0.0)

    rest_diff: float
    win_rate_diff: float
    elo_diff: float
    pts_diff_5: float
    pts_diff_15: float
    ast_diff_5: float
    ast_diff_15: float
    reb_diff_5: float
    reb_diff_15: float
    fg_pct_diff_5: float
    fg_pct_diff_15: float
    ft_pct_diff_5: float
    ft_pct_diff_15: float
    fg3_pct_diff_5: float
    fg3_pct_diff_15: float

    @field_validator("home_team", "visitor_team", mode="before")
    @classmethod
    def normalise_team_code(cls, v: str) -> str:
        """Normalize team code to uppercase."""
        return v.strip().upper()

    def to_feature_array(self) -> np.ndarray:
        """Convert request features to numpy array for model input."""
        return np.array(
            [getattr(self, col.lower()) for col in FEATURE_COLUMNS], dtype=float
        ).reshape(1, -1)

    def to_feature_dict(self) -> dict[str, float]:
        """Convert features to dictionary for drift detection."""
        return {
            col.lower(): float(getattr(self, col.lower())) for col in FEATURE_COLUMNS
        }


class PredictionResponse(BaseModel):
    """Response model for game prediction."""

    prediction: str
    confidence: float
    probability: dict[str, float]
    game_id: str
    processed_at: str
