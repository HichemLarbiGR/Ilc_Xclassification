"""Évaluation globale d'un modèle AttentionMIL sur un split figé.

Le script charge un modèle entraîné, prédit sur toutes les lames (train+val et
jeux de test), puis produit dans ``CONFIG.output_dir`` :

* les métriques par groupe (estimation ponctuelle + IC 95 % par bootstrap) ;
* une matrice de confusion et une courbe ROC (SVG) par jeu de test ;
* un CSV de toutes les prédictions et un CSV des erreurs triées par confiance.

Attention : les métriques train+val sont optimistes (le modèle a été entraîné
dessus). Seuls les jeux de test estiment la performance de généralisation.
"""
import logging
import os
import sys

import numpy as np
import torch

from models.MIL_models import build_model_class
from dataloader.data_loader_bag import TCGA_BRCA_BagDataset

from dataclass.config_class import TrainingConfig
from utils.utils import (
    _build_dict_report,
    _build_filename_figures,
    _build_files_figures_path,
    _build_path_results,
    _build_title_figures,
    _get_ids_from_model_path,
    _log_errors_predictions,
    build_name_h5_path,
    load_json,
    set_seed,
)
from utils.figures import plot_confusion_matrix, plot_roc_curve
from utils.reports_predictions import build_predictions_df
from training.training_utils import evaluate_model_tta
from training.metrics import get_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "abmil_risk.yaml")
# Clés du JSON de split évaluées comme jeux de test. Pour réactiver les jeux
# externes : ("test", "test_genova", "test_500"). L'ordre fixe la priorité en
# cas de chevauchement (une lame appartient au premier jeu qui la contient).


# ----------------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------------
def main() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CONFIG_PATH = os.path.join(
        PROJECT_ROOT,
        "config",
        "abmil_risk.yaml",
    )

    config = TrainingConfig.from_yaml(CONFIG_PATH)

    set_seed(config.seed)
    TEST_KEYS = config.subset_name

    # Load the predefined global data split.
    try:
        global_split = load_json(config.json_split_wsi_target_path)
    except FileNotFoundError:
        logger.error(
            "Global split file not found: %s. "
            "Run create_global_split.py first.",
            config.json_split_wsi_target_path,
        )
        sys.exit(1)

    logger.info(
        "Global split loaded from: %s",
        config.json_split_wsi_target_path,
    )

    if len(config.classes_name) != config.num_classes:
        raise ValueError(f"CLASS_NAMES ({len(config.classes_name)}) != CONFIG.num_classes ({config.num_classes})")

    os.makedirs(config.path_save_results, exist_ok=True)

    # --- 1. Split figé et labels --------------------------------------------

    train_val_slides = global_split["train_val"]

    test_slides = global_split[TEST_KEYS]

    logger.info("Split chargé")

    path_h5 = build_name_h5_path(config.h5_dir, config.foundation_model_name)

    # --- 2. Dataset et modèle ------------------------------------------------
    dataset_train = TCGA_BRCA_BagDataset(
        h5_dir=path_h5,
        slides=train_val_slides,
        is_train=False,
    )

    dataset_test = TCGA_BRCA_BagDataset(
        h5_dir=path_h5,
        slides=test_slides,
        is_train=False,
    )

    architecture_model = build_model_class(config.architecture_name)

    model = architecture_model(
        num_classes=config.num_classes,
        in_dim=config.input_dim,
    ).to(device)

    model.load_state_dict(
        torch.load(
            config.infer_model,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()

    logger.info("Modèle chargé")

    # --- 3. Prédictions (une seule passe sur toutes les lames) --------------
    _, _, _, y_train, pred_train, prob_train, ids_train = evaluate_model_tta(
        model,
        dataset_train,
        device,
        config,
    )

    _, _, _, y_test, pred_test, prob_test, ids_test = evaluate_model_tta(
        model,
        dataset_test,
        device,
        config,
    )

    y_train, y_test, pred_train, pred_test, prob_train, prob_test = (
        np.asarray(y_train),
        np.asarray(y_test),
        np.asarray(pred_train),
        np.asarray(pred_test),
        np.asarray(prob_train),
        np.asarray(prob_test),
    )

    # --- 5. Métriques et figures --------------------------------------------
    logger.info("=" * 60)

    model_id_value = _get_ids_from_model_path(config.infer_model)

    metrics_test = get_metrics(
        y_test,
        pred_test,
        prob_test,
        config.num_classes,
    )

    path_results = _build_path_results(config.path_save_results, model_id_value)
    os.makedirs(path_results, exist_ok=True)

    file_name_cm = _build_filename_figures(
        "confusion_matrix",
        config.architecture_name,
        config.foundation_model_name,
        model_id_value,
        config.figures_extension,
    )
    title_name_cm = _build_title_figures(
        "Confusion matrix",
        config.foundation_model_name,
        config.architecture_name,
        model_id_value,
    )
    path_figure_cm = _build_files_figures_path(path_results, file_name_cm)

    plot_confusion_matrix(
        metrics_test["confusion_matrix"],
        config.classes_name,
        title_name_cm,
        path_figure_cm,
    )

    file_name_roc = _build_filename_figures(
        "roc_curve",
        config.architecture_name,
        config.foundation_model_name,
        model_id_value,
        config.figures_extension,
    )
    title_name_roc = _build_title_figures(
        "ROC curve",
        config.foundation_model_name,
        config.architecture_name,
        model_id_value,
    )
    path_figure_roc = _build_files_figures_path(path_results, file_name_roc)

    plot_roc_curve(
        y_test,
        prob_test,
        config.classes_name,
        title_name_roc,
        path_figure_roc,
    )

    # --- 6. Tableaux de prédictions et d'erreurs ----------------------------

    dict_train_report = _build_dict_report("train_val", ids_train, y_train, pred_train, prob_train)
    dict_test_report = _build_dict_report(TEST_KEYS, ids_test, y_test, pred_test, prob_test)
    dict_report = {
        **dict_train_report,
        **dict_test_report,
    }

    df = build_predictions_df(
        dict_report,
        config.classes_name,
    )

    prob_cols = [c for c in df.columns if c.startswith("P_")]

    full_path = os.path.join(path_results, f"previsions_completes_{model_id_value}.csv")

    df.sort_values(["Dataset_Split", "Slide_ID"]).to_csv(full_path, index=False)

    logger.info("Prédictions complètes : %s", full_path)

    errors = df[df["Is_Error"]].sort_values("Confiance_Modele", ascending=False)

    errors_path = os.path.join(path_results, f"autopsie_erreurs_global_{model_id_value}.csv")

    errors.to_csv(errors_path, index=False)

    logger.info("Erreurs : %d sur %d lames (%s)", len(errors), len(df), errors_path)

    _log_errors_predictions(errors, prob_cols)


if __name__ == "__main__":
    main()