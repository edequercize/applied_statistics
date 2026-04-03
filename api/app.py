"""API FastAPI pour exposer le modèle de classification d'obésité."""

import logging
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import HealthResponse, PatientFeatures, PredictionResponse
from src.utils import INVERSE_MAPPING, ORDINAL_MAPPING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Obesity Classification API",
    description="API de prédiction du niveau d'obésité à partir de caractéristiques patient.",
    version="0.1.0",
)

# Monitoring Prometheus
Instrumentator().instrument(app).expose(app)

# ── Chargement du modèle ─────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
model = None


@app.on_event("startup")
def load_model():
    """Charge le modèle au démarrage de l'API."""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Modèle chargé depuis %s", MODEL_PATH)
    except FileNotFoundError:
        logger.warning("! Modèle non trouvé à %s — l'API démarrera sans modèle.", MODEL_PATH)


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check de l'API."""
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    """Prédit le niveau d'obésité pour un patient donné."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    # Conversion en DataFrame (1 ligne)
    input_data = pd.DataFrame([patient.model_dump()])

    try:
        prediction = model.predict(input_data)
        pred_value = prediction[0]

        # Gestion selon le type de modèle (ordinal int ou label string)
        if isinstance(pred_value, int | float):
            pred_ord = int(pred_value)
            pred_label = INVERSE_MAPPING.get(pred_ord, f"unknown_{pred_ord}")
        else:
            pred_label = str(pred_value)
            pred_ord = ORDINAL_MAPPING.get(pred_label, -1)

        logger.info("Prédiction : %s (ordinal=%d)", pred_label, pred_ord)
        return PredictionResponse(prediction=pred_label, prediction_ordinal=pred_ord)

    except Exception as e:
        logger.error("Erreur de prédiction : %s", e)
        raise HTTPException(status_code=500, detail=str(e))
