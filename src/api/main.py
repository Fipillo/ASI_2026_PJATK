import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from src.api.config import LOG_PATH
from src.api.schemas import GamePredictionRequest, PredictionResponse
from src import model_loader
from src.monitoring.drift_detector import DriftDetector

logger = logging.getLogger(__name__)
drift_detector = DriftDetector()


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


@app.get("/health")
async def health_check() -> dict[str, str]:
    model_status = "loaded" if model_loader.is_model_loaded() else "not_loaded"
    return {
        "status": "ok",
        "model_status": model_status,
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(request: GamePredictionRequest) -> PredictionResponse:
    logger.info("POST /predict  home=%s  visitor=%s  date=%s", request.home_team, request.visitor_team, request.game_date)
    
    # Extract features as dictionary
    feature_array = request.to_feature_array()
    feature_dict = request.to_feature_dict()
    
    # Check for data drift
    drift_result = drift_detector.check_drift(feature_dict)
    if drift_result["num_drifted"] > 0:
        logger.warning(
            "Data drift detected: %d/%d features drifted (ratio: %.2f)",
            drift_result["num_drifted"],
            drift_result["total_features"],
            drift_result["drift_ratio"]
        )
    
    try:
        prediction_class, probabilities = model_loader.predict(feature_array)
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

    logger.info("game_id=%s  result=%s  drift_ratio=%.2f", game_id, prediction_label, drift_result["drift_ratio"])

    return PredictionResponse(
        prediction=prediction_label,
        confidence=home_prob if prediction_class == 1 else visitor_prob,
        probability={"home_team_wins": home_prob, "visitor_team_wins": visitor_prob},
        game_id=game_id,
        processed_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/api/v1/drift")
async def get_drift_metrics() -> dict:
    """Get current drift detection statistics."""
    return {
        "reference_stats_loaded": bool(drift_detector.reference_stats),
        "num_reference_features": len(drift_detector.reference_stats),
        "drift_stats_file": str(drift_detector.stats_file),
    }
