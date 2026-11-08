"""
Bootstrap-based confidence interval estimation.

This module provides utilities for estimating confidence
intervals for classification metrics.
"""

from typing import Tuple, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _sensitivity_from_cm(cm: np.ndarray) -> float:
    """Sensitivity (TPR / recall) for binary confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    return tp / (tp + fn) if (tp + fn) else np.nan


def _specificity_from_cm(cm: np.ndarray) -> float:
    """Specificity (TNR) for binary confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) else np.nan


def _compute_metric(
    metric_name: str,
    y_true_bs: np.ndarray,
    y_pred_bs: np.ndarray,
    y_proba_bs: Optional[np.ndarray],
    n_classes: int,
) -> float:
    """
    Compute a single metric on a bootstrap sample.
    Returns np.nan when the metric is undefined on that sample.
    """
    # ---- Metrics that only need y_true / y_pred -------------------------- #
    if metric_name == "accuracy":
        return accuracy_score(y_true_bs, y_pred_bs)

    if metric_name == "f1_macro":
        return f1_score(y_true_bs, y_pred_bs, average="macro", zero_division=0)

    if metric_name == "f1_weighted":
        return f1_score(y_true_bs, y_pred_bs, average="weighted", zero_division=0)

    if metric_name == "precision_macro":
        return precision_score(y_true_bs, y_pred_bs, average="macro", zero_division=0)

    if metric_name == "recall_macro":
        return recall_score(y_true_bs, y_pred_bs, average="macro", zero_division=0)

    if metric_name in ("sensitivity", "specificity"):
        if n_classes != 2:
            return np.nan
        cm = confusion_matrix(y_true_bs, y_pred_bs, labels=[0, 1])
        if cm.shape != (2, 2):
            return np.nan
        return (
            _sensitivity_from_cm(cm)
            if metric_name == "sensitivity"
            else _specificity_from_cm(cm)
        )

    # ---- Metrics that need probabilities ---------------------------------- #
    if y_proba_bs is None:
        return np.nan

    # AUC / AUPRC need at least 2 classes in the bootstrap sample.
    if len(np.unique(y_true_bs)) < 2:
        return np.nan

    try:
        if metric_name == "auc":
            if n_classes == 2:
                return roc_auc_score(y_true_bs, y_proba_bs[:, 1])
            y_bin = label_binarize(y_true_bs, classes=list(range(n_classes)))
            return roc_auc_score(
                y_bin, y_proba_bs, multi_class="ovr", average="macro"
            )

        if metric_name == "auprc":
            if n_classes == 2:
                return average_precision_score(y_true_bs, y_proba_bs[:, 1])
            y_bin = label_binarize(y_true_bs, classes=list(range(n_classes)))
            return average_precision_score(y_bin, y_proba_bs, average="macro")

    except ValueError:
        return np.nan

    return np.nan


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_class: np.ndarray,
    metric_name: str,
    n_iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute a bootstrap confidence interval for a classification metric.

    Supported `metric_name` values:
        - 'accuracy'
        - 'precision_macro'
        - 'recall_macro'
        - 'f1_macro'
        - 'f1_weighted'
        - 'sensitivity'      (binary only)
        - 'specificity'      (binary only)
        - 'auc'              (binary or multiclass OVR macro)
        - 'auprc'            (binary or multiclass OVR macro)

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred_proba : np.ndarray
        Predicted probabilities for each class (may be None for metrics
        that do not require probabilities).
    y_pred_class : np.ndarray
        Predicted class labels.
    metric_name : str
        Name of the metric to evaluate.
    n_iterations : int, default=2000
        Number of bootstrap iterations.
    alpha : float, default=0.05
        Confidence level for the bootstrap interval (1 - alpha).
    seed : int, default=42
        Seed for reproducibility.

    Returns
    -------
    Tuple[float, float, float]
        (mean bootstrap score, lower bound, upper bound).
        Returns (nan, nan, nan) if no valid score is produced.
    """
    supported = {
        "accuracy", "precision_macro", "recall_macro",
        "f1_macro", "f1_weighted",
        "sensitivity", "specificity",
        "auc", "auprc",
    }
    if metric_name not in supported:
        raise ValueError(
            f"Unsupported metric '{metric_name}'. "
            f"Supported: {sorted(supported)}"
        )

    rng = np.random.default_rng(seed)
    n_samples = len(y_true)

    n_classes = (
        y_pred_proba.shape[1]
        if y_pred_proba is not None and y_pred_proba.ndim == 2
        else len(np.unique(y_true))
    )

    scores = []
    for _ in range(n_iterations):
        indices = rng.choice(n_samples, size=n_samples, replace=True)

        y_true_bs = y_true[indices]
        y_pred_bs = y_pred_class[indices]
        y_proba_bs = (
            y_pred_proba[indices] if y_pred_proba is not None else None
        )

        scores.append(
            _compute_metric(
                metric_name, y_true_bs, y_pred_bs, y_proba_bs, n_classes
            )
        )

    valid_scores = [s for s in scores if not np.isnan(s)]
    if not valid_scores:
        return np.nan, np.nan, np.nan

    lower = float(np.percentile(valid_scores, alpha / 2 * 100))
    upper = float(np.percentile(valid_scores, (1 - alpha / 2) * 100))
    return float(np.mean(valid_scores)), lower, upper