"""Point d'entrée principal — Entraînement LightGBM et tracking MLflow.

Usage
-----
    python main.py
    python main.py --config configs/params.yaml
"""

import argparse
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from src.data.load import load_data
from src.data.preprocess import (
    build_preprocessor,
    clean_ordinal_columns,
    split_data,
)
from src.models.evaluate import log_evaluation, plot_confusion_matrix
from src.models.train import train_lightgbm

from src.utils import ORDINAL_MAPPING, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def run_experiment(config: dict):
    """Lance l'entraînement LightGBM avec tracking MLflow."""
    cfg_data = config["data"]
    cfg_feat = config["features"]
    cfg_mlflow = config["mlflow"]
    cfg_lgb = config["model"]

    # ── Chargement et préparation ────────────────────────────────────────
    df = load_data(cfg_data["path"])
    df = clean_ordinal_columns(df, cfg_feat["ordinal"])

    X_train, X_test, y_train, y_test = split_data(
        df,
        target=cfg_data["target"],
        drop_columns=cfg_data.get("drop_columns"),
        test_size=cfg_data["test_size"],
        random_state=cfg_data["random_state"],
    )

    preprocessor = build_preprocessor(
        X_train,
        ord_cols=cfg_feat["ordinal"],
        ohe_min_frequency=cfg_feat["ohe_min_frequency"],
    )

    # ── MLflow setup ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg_mlflow["tracking_uri"])
    mlflow.set_experiment(cfg_mlflow["experiment_name"])

    # ── Entraînement LightGBM ────────────────────────────────────────────
    with mlflow.start_run(run_name="lightgbm"):
        mlflow.set_tag("model_type", "LightGBM")

        grid = train_lightgbm(
            X_train, y_train, preprocessor,
            param_grid=cfg_lgb["param_grid"],
            cv=cfg_lgb["cv"],
        )

        # Log des hyperparamètres et métriques
        mlflow.log_params(grid.best_params_)

        y_pred = grid.predict(X_test)
        metrics = log_evaluation(y_test, y_pred, "test")
        metrics.update(log_evaluation(y_train, grid.predict(X_train), "train"))
        metrics["best_cv_mae"] = -grid.best_score_
        mlflow.log_metrics(metrics)

        # Confusion matrix
        y_test_ord = y_test.map(ORDINAL_MAPPING)
        y_pred_ord = [ORDINAL_MAPPING.get(v, v) for v in y_pred]
        fig = plot_confusion_matrix(y_test_ord, y_pred_ord, "Confusion — LightGBM")
        mlflow.log_figure(fig, "confusion_matrix.png")

        # Sauvegarde du modèle
        mlflow.sklearn.log_model(grid.best_estimator_, "model")
        model_path = Path("models/best_model.joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(grid.best_estimator_, model_path)
        logger.info("✅ LightGBM — terminé. Modèle sauvegardé dans models/best_model.joblib")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entraînement LightGBM — Obesity Classification")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/params.yaml",
        help="Chemin vers le fichier de configuration YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
