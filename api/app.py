"""API FastAPI pour exposer le modèle de classification d'obésité."""

import logging
import os
from pathlib import Path

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


def _download_model_from_s3(s3_url: str, dest: Path) -> None:
    """Télécharge le modèle depuis MinIO/S3 vers le chemin local."""
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    parts = s3_url.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    endpoint = os.getenv("AWS_S3_ENDPOINT_URL")
    s3 = boto3.client("s3", endpoint_url=endpoint)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(bucket, key, str(dest))
        logger.info("Modèle téléchargé depuis %s", s3_url)
    except (BotoCoreError, ClientError) as e:
        logger.error("Échec téléchargement S3 : %s", e)


@app.on_event("startup")
def load_model():
    """Charge le modèle au démarrage de l'API, en le téléchargeant depuis S3 si nécessaire."""
    global model
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        s3_url = os.getenv("MODEL_S3_URL")
        if s3_url:
            logger.info("Modèle absent localement, téléchargement depuis %s", s3_url)
            _download_model_from_s3(s3_url, model_path)

    try:
        model = joblib.load(model_path)
        logger.info("Modèle chargé depuis %s", model_path)
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
