"""Constantes et fonctions utilitaires partagées."""

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
