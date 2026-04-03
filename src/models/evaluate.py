import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error

from src.utils import ORDINAL_MAPPING


def mae_ordinale(y_true, y_pred) -> float:
    """
    Mean absolute error on ordinal obesity classes.
    Accepts either labels or numeric inputs.
    """
    if len(y_true) == 0:
        return 0.0

    def convert(values):
        result = []
        for v in values:
            result.append(ORDINAL_MAPPING.get(v, v))
        return result

    y_true_num = convert(y_true)
    y_pred_num = convert(y_pred)

    return float(mean_absolute_error(y_true_num, y_pred_num))


def log_evaluation(y_true, y_pred, prefix: str = "test") -> dict:
    """
    Compute evaluation metrics for MLflow logging.
    """
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_ordinal_mae": mae_ordinale(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Create matplotlib confusion matrix figure.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    fig.colorbar(im, ax=ax)

    return fig