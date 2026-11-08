"""
Configuration and Model Logging Module
======================================

This module provides utilities for logging model configuration,
hyperparameters, architecture details, and parameter counts.

The logging output includes configuration settings, public model
attributes, the total number of model parameters, and the complete
model architecture.

Functions
---------
log_config_and_model
    Log the configuration, model attributes, parameter count,
    and architecture using the configured logger.
load_json
    Load and return the contents of a JSON file.
"""

import json
import logging
import math
import os
import random
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def log_config_and_model(
    config: Any,
    model: nn.Module,
    title: str = "CONFIGURATION",
) -> None:
    """
    Log the configuration and model architecture.

    Parameters
    ----------
    config : Any
        Configuration object represented by a dataclass.
    model : nn.Module
        PyTorch model to inspect and log.
    title : str, default="CONFIGURATION"
        Title displayed in the log output.

    Returns
    -------
    None
        This function writes the formatted output to the logger.
    """
    config_dict = asdict(config)
    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    lines = [
        "=" * 70,
        f"  {title}",
        "=" * 70,
        "",
        "  Hyperparameters:",
        "  " + "-" * 68,
    ]

    # Format configuration values.
    for key, value in config_dict.items():
        value_str = str(value)

        if len(value_str) > 55:
            value_str = "..." + value_str[-52:]

        lines.append(f"  {key:25s}: {value_str}")

    # Extract public scalar attributes from the model.
    model_attrs = {
        key: value
        for key, value in vars(model).items()
        if (
            not key.startswith("_")
            and not isinstance(value, nn.Module)
            and not isinstance(value, nn.Parameter)
            and type(value) in (int, float, str, bool)
        )
    }

    lines.extend(
        [
            "",
            f"  Architecture: {model.__class__.__name__}",
            "  " + "-" * 68,
            f"    Total parameters: {total_params:,}",
            "",
            "    Constructor attributes:",
        ]
    )

    for key, value in model_attrs.items():
        lines.append(f"      {key:20s}: {value}")

    lines.extend(
        [
            "",
            "    Model structure:",
            str(model),
            "",
            "=" * 70,
        ]
    )

    logger.info("\n".join(lines))


def load_json(path):
    """Load and return the contents of a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _clean(obj):
    """numpy -> python, NaN/inf -> None. Rien d'autre."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def save_run_json(run_dict, save_dir):
    """
    Écrit un JSON lisible depuis ton dict de résultats.

    Args:
        run_dict: dict avec 'Configuration', 'Fold 1'... 'Fold N', 'Test'.
        save_dir: dossier de sortie.

    Returns:
        Le chemin du JSON écrit.
    """
    os.makedirs(save_dir, exist_ok=True)

    run_id    = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    timestamp = datetime.now().isoformat(timespec="seconds")

    # --- Config ---
    config = run_dict["Configuration"]

    # --- Folds (ordre numérique, pas lexicographique) ---
    fold_keys = sorted(
        (k for k in run_dict if k.startswith("Fold ")),
        key=lambda k: int(k.split()[1]),
    )
    folds = []
    for k in fold_keys:
        f = run_dict[k]
        folds.append({
            "fold":        int(k.split()[1]),
            "final_epoch": f["Final epoch"],
            "loss": {
                "train": f["Train losses"],
                "val":   f["Val losses"],
            },
            "train": _clean(f["Train metrics"]),
            "val":   _clean(f["Val metrics"]),
        })

    # --- Test ---
    test = {}
    for metric, triple in run_dict["Test"].items():
        mean, low, high = triple
        test[metric] = {"mean": mean, "ci_low": low, "ci_high": high}

    record = {
        "run_id":    run_id,
        "timestamp": timestamp,
        "config":    _clean(config),
        "folds":     folds,
        "test":      _clean(test),
    }

    path_result = create_path_results(save_dir, run_id)

    create_folder_results(path_result)

    path = os.path.join(path_result, f"{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return run_id, timestamp, path_result


def build_name_model_pth(path, architecture, foundation_model_name, id_model, seed):
    return f"{path}/model_{architecture}_{foundation_model_name.lower()}_seed_{seed}_{id_model}.pth"


def build_name_h5_path(h5_dir, fm_model_name):
    return h5_dir + "/" + fm_model_name.upper()


def set_seed(seed: int) -> None:
    """Fixe les graines Python, NumPy et PyTorch (CPU et GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # sans effet s'il n'y a pas de GPU


def clean_label(value):
    """Convertit une valeur brute du JSON de labels en int, ou None si inutilisable.

    Accepte un scalaire ou une liste (on garde le premier élément) et rejette
    None / NaN, y compris un NaN placé dans une liste.
    """
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(value)


def log_scores(results: dict) -> None:
    """Affiche les métriques ; l'IC n'est affiché que s'il a été calculé."""
    for name, (value, low, high) in results.items():
        has_ci = (
            low is not None
            and high is not None
            and not (isinstance(low, float) and math.isnan(low))
            and not (isinstance(high, float) and math.isnan(high))
        )
        ci = (
            f"  [IC 95 % : {low:.4f} - {high:.4f}]"
            if has_ci
            else ""
        )
        logger.info("  %-16s: %.4f%s", name, value, ci)



def _log_errors_predictions(errors: pd.DataFrame, prob_cols: list[str]) -> None:

    logger.info("Top 15 des erreurs de test les plus confiantes (à relire) :")

    for _, row in errors[errors["Dataset_Split"] == "Test"].head(15).iterrows():
        probas = ", ".join(f"{c}:{row[c]:.2f}" for c in prob_cols)
        logger.info(
            "  Slide %d [%s] | Vrai: %d -> Prédit: %d | [%s]",
            row["Slide_ID"], row["Test_Subset"], row["True_Label"],
            row["Pred_Label"], probas,
        )


def _build_title_figures(basename: str, fm_name: str, archi_name: str, model_id_value: str) -> str:
    return f"{basename} ( {fm_name} - {archi_name} - {model_id_value} )"

def _get_ids_from_model_path(path: str) -> str:
    sans_extension = path.rsplit('.', 1)[0]
    parts = sans_extension.rsplit('_', 3)
    resultat = '_'.join(parts[-3:])
    return resultat


def create_folder_results(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    return


def create_path_results(s_dir, r_id):
    return f"{s_dir}/{r_id}"


def _build_path_results(path: str, r_id: str) -> str:
    return f"{path}/{r_id}"


def _build_files_figures_path(path:str, file_name: str):
    return f"{path}/{file_name}"


def _build_dict_report(split, ids_list, y_list, pred_list, prob_list):
    return {
        split: [
            (ids_list[i], y_list[i], pred_list[i], prob_list[i])
            for i in range(len(ids_list))
        ]
    }

def _build_filename_figures(basename:str, arch_model:str,fm_model:str, model_id_value:str, ext:str):
    return f"{basename}_{arch_model}_{fm_model}_{model_id_value}.{ext}"

