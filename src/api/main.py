import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from src.api.config import LOG_PATH
from src.api.schemas import GamePredictionRequest, PredictionResponse
from src import model_loader

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
    )
    try:
        model_loader.load()
    except FileNotFoundError as exc:
        logger.error("Could not load model: %s", exc)
    yield


app = FastAPI(title="NBA Game Prediction API", version="1.0.0", lifespan=lifespan)


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(request: GamePredictionRequest) -> PredictionResponse:
    logger.info("POST /predict  home=%s  visitor=%s  date=%s", request.home_team, request.visitor_team, request.game_date)
    try:
        prediction_class, probabilities = model_loader.predict(request.to_feature_array())
    except RuntimeError as exc:
        logger.error("Model not loaded: %s", exc)
        raise HTTPException(status_code=500, detail="Model is not available.")
    except Exception as exc:
        logger.error("Inference error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed.")

    home_prob = round(float(probabilities[1]), 4)
    visitor_prob = round(float(probabilities[0]), 4)
    prediction_label = "home_team_wins" if prediction_class == 1 else "visitor_team_wins"
    game_id = f"{request.game_date.strftime('%Y%m%d')}_{request.home_team}_{request.visitor_team}"

    logger.info("game_id=%s  result=%s", game_id, prediction_label)

    return PredictionResponse(
        prediction=prediction_label,
        confidence=home_prob if prediction_class == 1 else visitor_prob,
        probability={"home_team_wins": home_prob, "visitor_team_wins": visitor_prob},
        game_id=game_id,
        processed_at=datetime.now(timezone.utc).isoformat(),
    )
