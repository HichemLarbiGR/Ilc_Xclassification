"""
Bootstrap-based confidence interval estimation.

This module provides utilities for estimating confidence
intervals for classification metrics.
"""

from typing import Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_class: np.ndarray,
    metric_name: str,
    n_iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute a bootstrap confidence interval for a classification metric.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred_proba : np.ndarray
        Predicted probabilities for each class.
    y_pred_class : np.ndarray
        Predicted class labels.
    metric_name : str
        Name of the metric to evaluate. Supported metrics are
        'accuracy', 'f1_macro', and 'auc'.
    n_iterations : int, default=2000
        Number of bootstrap iterations.
    alpha : float, default=0.05
        Confidence level for the bootstrap interval.

    Returns
    -------
    Tuple[float, float, float]
        A tuple containing the mean bootstrap score, the lower
        confidence bound, and the upper confidence bound.
        Returns (NaN, NaN, NaN) if no valid scores are available.
    """
    rng = np.random.default_rng(seed)
    scores = []
    n_samples = len(y_true)

    n_classes = (
        y_pred_proba.shape[1]
        if y_pred_proba is not None and y_pred_proba.ndim == 2
        else len(np.unique(y_true))
    )

    # Generate bootstrap samples with replacement.
    for _ in range(n_iterations):
        indices = rng.choice(
            n_samples, 
            size=n_samples, 
            replace=True)

        y_true_bs = y_true[indices]
        y_pred_bs = y_pred_class[indices]

        # Compute the selected classification metric.
        if metric_name == "accuracy":
            scores.append(
                accuracy_score(y_true_bs, y_pred_bs)
            )

        elif metric_name == "f1_macro":
            scores.append(
                f1_score(
                    y_true_bs,
                    y_pred_bs,
                    average="macro",
                    zero_division=0,
                )
            )
        elif metric_name == "recall_macro":
            scores.append(
                recall_score(
                    y_true_bs,
                    y_pred_bs,
                    average="macro",
                    zero_division=0,
                )
            )

        elif metric_name == "precision_macro":
            scores.append(
                precision_score(
                    y_true_bs,
                    y_pred_bs,
                    average="macro",
                    zero_division=0,
                )
            )
        elif metric_name == "auc" and y_pred_proba is not None:
            
            y_proba_bs = y_pred_proba[indices]

            # AUC requires at least two classes in the bootstrap sample.
            if len(np.unique(y_true_bs)) < 2:
                scores.append(np.nan)
                continue

            try:
                if n_classes == 2:
                    # Binary case: use proba of the positive class, 1D labels.
                    scores.append(roc_auc_score(y_true_bs, y_proba_bs[:, 1]))
                else:
                    y_true_bs_bin = label_binarize(y_true_bs, classes=list(range(n_classes)))
                    scores.append(roc_auc_score(
                        y_true_bs_bin, y_proba_bs,
                        multi_class="ovr", average="macro",
                    ))
            except ValueError as e:
                # Optional: log to see if it ever triggers.
                # print(f"[bootstrap_ci][auc] {type(e).__name__}: {e}")
                scores.append(np.nan)

    # Exclude invalid scores before computing the confidence interval.
    valid_scores = [
        score for score in scores
        if not np.isnan(score)
    ]

    if not valid_scores:
        return np.nan, np.nan, np.nan

    # Compute the mean score and the confidence interval bounds.
    lower = np.percentile(
        valid_scores,
        alpha / 2 * 100
    )
    upper = np.percentile(
        valid_scores, 
        (1 - alpha / 2) * 100
        )

    return np.mean(valid_scores), lower, upper
