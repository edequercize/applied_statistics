"""Constantes et fonctions utilitaires partagées."""

import numpy as np
import yaml


# ── Mapping ordinal des classes ──────────────────────────────────────────────

ORDINAL_MAPPING = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 1,
    "Overweight_Level_I": 2,
    "Overweight_Level_II": 3,
    "Obesity_Type_I": 4,
    "Obesity_Type_II": 5,
    "Obesity_Type_III": 6,
}

INVERSE_MAPPING = {v: k for k, v in ORDINAL_MAPPING.items()}


# ── Chargement de la configuration ───────────────────────────────────────────

def load_config(path: str = "configs/params.yaml") -> dict:
    """Charge le fichier de configuration YAML."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Encodage CORAL ───────────────────────────────────────────────────────────

def coral_encode(y, n_classes: int = 7) -> np.ndarray:
    """Encode des labels ordinaux (0…K-1) en matrice binaire CORAL (n, K-1).

    Pour chaque échantillon, les colonnes j < label valent 1, les autres 0.
    """
    y = np.asarray(y)
    return np.greater_equal.outer(y, np.arange(1, n_classes)).astype("float32")


def coral_decode(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convertit les probabilités CORAL en classes prédites."""
    return (proba > threshold).sum(axis=1)
