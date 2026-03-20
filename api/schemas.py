"""Schémas Pydantic pour l'API de prédiction."""

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """Features d'entrée pour la prédiction du niveau d'obésité."""

    Gender: str = Field(..., example="Male")
    Age: float = Field(..., example=25.0)
    family_history_with_overweight: str = Field(..., example="yes")
    FAVC: str = Field(..., example="yes")
    FCVC: int = Field(..., ge=1, le=3, example=2)
    NCP: int = Field(..., ge=1, le=4, example=3)
    CAEC: str = Field(..., example="Sometimes")
    SMOKE: str = Field(..., example="no")
    CH2O: int = Field(..., ge=1, le=3, example=2)
    SCC: str = Field(..., example="no")
    FAF: int = Field(..., ge=0, le=3, example=1)
    TUE: int = Field(..., ge=0, le=2, example=1)
    CALC: str = Field(..., example="Sometimes")
    MTRANS: str = Field(..., example="Public_Transportation")


class PredictionResponse(BaseModel):
    """Réponse de l'API avec la prédiction."""

    prediction: str = Field(..., example="Obesity_Type_I")
    prediction_ordinal: int = Field(..., example=4)


class HealthResponse(BaseModel):
    """Réponse du health check."""

    status: str = "ok"
    model_loaded: bool = True
