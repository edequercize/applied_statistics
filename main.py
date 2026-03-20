"""Point d'entrée principal — Entraînement et tracking MLflow.

Usage
-----
    python main.py --model lightgbm --config configs/params.yaml
    python main.py --model all
"""

import argparse
import logging

import joblib
import mlflow
import mlflow.sklearn

from src.data.load import load_data
from src.data.preprocess import (
    build_preprocessor,
    clean_ordinal_columns,
    split_data,
)
from src.models.evaluate import log_evaluation, mae_ordinale, plot_confusion_matrix
from src.models.train import (
    predict_coral_nn,
    train_coral_nn,
    train_lightgbm,
    train_mord,
    train_random_forest,
)
from src.utils import ORDINAL_MAPPING, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def run_experiment(model_name: str, config: dict):
    """Lance l'entraînement d'un modèle avec tracking MLflow."""
    cfg_data = config["data"]
    cfg_feat = config["features"]
    cfg_mlflow = config["mlflow"]

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

    # ── Entraînement selon le modèle choisi ──────────────────────────────
    if model_name in ("mord", "all"):
        _run_mord(X_train, X_test, y_train, y_test, preprocessor)

    if model_name in ("random_forest", "all"):
        cfg_rf = config["models"]["random_forest"]
        _run_rf(X_train, X_test, y_train, y_test, preprocessor, cfg_rf)

    if model_name in ("lightgbm", "all"):
        cfg_lgb = config["models"]["lightgbm"]
        _run_lgb(X_train, X_test, y_train, y_test, preprocessor, cfg_lgb)

    if model_name in ("coral", "all"):
        cfg_coral = config["models"]["coral"]
        _run_coral(X_train, X_test, y_train, y_test, preprocessor, cfg_coral)


# ── Fonctions internes par modèle ────────────────────────────────────────────


def _run_mord(X_train, X_test, y_train, y_test, preprocessor):
    with mlflow.start_run(run_name="mord_logistic_it"):
        mlflow.set_tag("model_type", "mord_LogisticIT")

        pipe = train_mord(X_train, y_train, preprocessor)

        y_test_ord = y_test.map(ORDINAL_MAPPING)
        y_pred = pipe.predict(X_test)

        metrics = log_evaluation(y_test_ord, y_pred, "test")
        metrics.update(log_evaluation(y_train.map(ORDINAL_MAPPING), pipe.predict(X_train), "train"))
        mlflow.log_metrics(metrics)

        fig = plot_confusion_matrix(y_test_ord, y_pred, "Confusion — mord LogisticIT")
        mlflow.log_figure(fig, "confusion_matrix.png")

        mlflow.sklearn.log_model(pipe, "model")
        logger.info("✅ mord — terminé")


def _run_rf(X_train, X_test, y_train, y_test, preprocessor, cfg):
    with mlflow.start_run(run_name="random_forest"):
        mlflow.set_tag("model_type", "RandomForest")

        grid = train_random_forest(
            X_train, y_train, preprocessor,
            param_grid=cfg["param_grid"],
            cv=cfg["cv"],
        )

        mlflow.log_params(grid.best_params_)
        y_pred = grid.predict(X_test)
        metrics = log_evaluation(y_test, y_pred, "test")
        metrics.update(log_evaluation(y_train, grid.predict(X_train), "train"))
        metrics["best_cv_mae"] = -grid.best_score_
        mlflow.log_metrics(metrics)

        y_test_ord = y_test.map(ORDINAL_MAPPING)
        y_pred_ord = y_pred if y_pred.dtype in ("int64", "int32") else [ORDINAL_MAPPING.get(v, v) for v in y_pred]
        fig = plot_confusion_matrix(y_test_ord, y_pred_ord, "Confusion — Random Forest")
        mlflow.log_figure(fig, "confusion_matrix.png")

        mlflow.sklearn.log_model(grid.best_estimator_, "model")
        logger.info("✅ Random Forest — terminé")


def _run_lgb(X_train, X_test, y_train, y_test, preprocessor, cfg):
    with mlflow.start_run(run_name="lightgbm"):
        mlflow.set_tag("model_type", "LightGBM")

        grid = train_lightgbm(
            X_train, y_train, preprocessor,
            param_grid=cfg["param_grid"],
            cv=cfg["cv"],
        )

        mlflow.log_params(grid.best_params_)
        y_pred = grid.predict(X_test)
        metrics = log_evaluation(y_test, y_pred, "test")
        metrics.update(log_evaluation(y_train, grid.predict(X_train), "train"))
        metrics["best_cv_mae"] = -grid.best_score_
        mlflow.log_metrics(metrics)

        y_test_ord = y_test.map(ORDINAL_MAPPING)
        y_pred_ord = y_pred if y_pred.dtype in ("int64", "int32") else [ORDINAL_MAPPING.get(v, v) for v in y_pred]
        fig = plot_confusion_matrix(y_test_ord, y_pred_ord, "Confusion — LightGBM")
        mlflow.log_figure(fig, "confusion_matrix.png")

        mlflow.sklearn.log_model(grid.best_estimator_, "model")
        logger.info("✅ LightGBM — terminé")


def _run_coral(X_train, X_test, y_train, y_test, preprocessor, cfg):
    with mlflow.start_run(run_name="coral_nn"):
        mlflow.set_tag("model_type", "CORAL_NN")

        # Le preprocessor doit être fit avant CORAL
        preprocessor.fit(X_train)

        mlflow.log_params({
            "hidden_layers": cfg["hidden_layers"],
            "dropout": cfg["dropout"],
            "learning_rate": cfg["learning_rate"],
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
        })

        model, history = train_coral_nn(
            X_train, y_train, preprocessor,
            hidden_layers=cfg["hidden_layers"],
            dropout=cfg["dropout"],
            learning_rate=cfg["learning_rate"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
        )

        y_test_ord = y_test.map(ORDINAL_MAPPING)
        y_pred = predict_coral_nn(model, X_test, preprocessor)

        metrics = log_evaluation(y_test_ord, y_pred, "test")
        mlflow.log_metrics(metrics)

        fig = plot_confusion_matrix(y_test_ord, y_pred, "Confusion — CORAL NN")
        mlflow.log_figure(fig, "confusion_matrix.png")

        # Sauvegarder le modèle Keras + preprocessor
        model.save("models/coral_nn.keras")
        joblib.dump(preprocessor, "models/preprocessor.joblib")
        mlflow.log_artifacts("models", "model")

        logger.info("✅ CORAL NN — terminé")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entraînement des modèles d'obésité")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["mord", "random_forest", "lightgbm", "coral", "all"],
        help="Modèle à entraîner (default: all)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/params.yaml",
        help="Chemin vers le fichier de configuration YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(args.model, config)


if __name__ == "__main__":
    main()
