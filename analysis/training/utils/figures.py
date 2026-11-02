from __future__ import annotations

import itertools
import logging

import matplotlib
matplotlib.use("Agg")                     # serveur sans display
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title_name:str,
    path_figure: str,
) -> None:
    """Enregistre une matrice de confusion (effectifs entiers) au format SVG."""
    # Forcer un array numérique (évite dtype=object -> TypeError dans imshow)
    cm = np.asarray(cm, dtype=np.float64)
    
    
    
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names)

    ax.set_title(title_name)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j, i, f"{int(round(cm[i, j]))}",
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    fig.tight_layout()
    fig.savefig(path_figure, format="svg", bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure enregistrée : %s", path_figure)


def plot_roc_curve(
    y_true,
    y_prob,
    class_names: list[str],
    title_name: str,
    path_figure: str,
) -> None:
    """Enregistre la ROC : une courbe en binaire, une par classe + micro-moyenne sinon."""

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(8, 6))

    if n == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        ax.plot(fpr, tpr, lw=2,
                label=f"ROC {class_names[1]} (AUC = {auc(fpr, tpr):.2f})")
    else:
        y_bin = label_binarize(y_true, classes=list(range(n)))
        for i, name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, lw=2,
                    label=f"ROC {name} (AUC = {auc(fpr, tpr):.2f})")
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
        ax.plot(fpr, tpr, ls=":", lw=4, color="deeppink",
                label=f"Micro-averaged (AUC = {auc(fpr, tpr):.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set(
        xlim=(0, 1), ylim=(0, 1.05),
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=title_name,
    )

    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path_figure, format="svg", bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure enregistrée : %s", path_figure)