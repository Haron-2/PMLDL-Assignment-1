"""Stage 4 - FastAPI model-serving endpoint.

Serves the single sklearn Pipeline artifact (Stage 3) that goes from RAW
features to a prediction - no preprocessing logic is duplicated here.

Endpoints:
    GET  /         service info
    GET  /health   liveness + model load status (always HTTP 200)
    POST /predict  wine features (JSON, dataset column names) -> prediction

Error contract:
    422  request validation failure (types, ranges, free>total SO2, wine_type)
    503  model artifact not loaded / not found
    500  unexpected prediction failure

Environment:
    MODEL_PATH     override artifact location (used by the Docker image)
    MODEL_VERSION  label reported in responses (default: rf-300-baseline)

Run locally (from project root):
    .venv/Scripts/python.exe -m uvicorn main:app --app-dir code/deployment/api --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from schemas import HealthResponse, PredictionResponse, WineFeatures

API_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = API_DIR.parents[2] / "models" / "model.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
MODEL_VERSION = os.getenv("MODEL_VERSION", "rf-300-baseline")

_state: dict = {"model": None}


def _load_model() -> None:
    _state["model"] = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="Wine Quality Prediction API",
    description="Predicts binary wine quality (Good: quality > 5) from physicochemical tests "
                "for red and white wines (UCI Wine Quality, merged).",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/")
async def root() -> dict:
    return {
        "service": "wine-quality-prediction-api",
        "version": MODEL_VERSION,
        "endpoints": {"health": "GET /health", "predict": "POST /predict", "docs": "GET /docs"},
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    model = _state["model"]
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_path=str(MODEL_PATH),
        model_version=MODEL_VERSION,
        model_type=type(model).__name__ if model is not None else None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: WineFeatures) -> PredictionResponse:
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model artifact not loaded")
    try:
        frame = pd.DataFrame([features.model_dump(by_alias=True)])
        proba_good = float(model.predict_proba(frame)[0][1])
        prediction = int(model.predict(frame)[0])
    except Exception as exc:  # noqa: BLE001 - converted to HTTP 500 contract
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
    return PredictionResponse(
        prediction=prediction,
        label="Good" if prediction == 1 else "Poor",
        probability_good=round(proba_good, 4),
        model_version=MODEL_VERSION,
    )
