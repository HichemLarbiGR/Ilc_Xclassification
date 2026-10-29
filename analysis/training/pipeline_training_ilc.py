"""Main training script for multiple-instance learning (MIL).

This script loads a predefined global data split, filters slides with
valid clinical labels, initializes the datasets, and runs stratified
K-fold cross-validation followed by final model training and test
evaluation.
"""

import json
import logging
import math
import os
import random
import sys
import warnings
from collections import Counter

import numpy as np
import torch

# Add the project root to the Python module search path.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.MIL_models import AttentionMIL
from dataloader.data_loader_bag import TCGA_BRCA_BagDataset

from dataclass.config_class import TrainingConfig
from utils.utils import log_config_and_model, load_json
from training.training_utils import run_stratified_kfold_training


# Logging configuration.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# Reproducibility configuration.
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def main():
    """Run dataset preparation and the complete training pipeline."""
    
    CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "training/config",
    "abmil_risk.yaml",
        )
    print(CONFIG_PATH)
    config = TrainingConfig.from_yaml(CONFIG_PATH)

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

    # Load the global split (id, filename, target).
    global_split = load_json(config.json_split_wsi_target_path)

    train_val_slides = global_split["train_val"] 
    test_slides      = global_split["test"]        

    train_dist_before = dict(Counter(s["target"] for s in train_val_slides))
    test_dist_before  = dict(Counter(s["target"] for s in test_slides))

    total_global = len(train_val_slides) + len(test_slides)

    # Initialize the datasets.
    try:
        logger.info(
            "Initializing datasets and filtering slides "
        )

        train_dataset = TCGA_BRCA_BagDataset(
            h5_dir=config.h5_dir,
            slides=train_val_slides,
            is_train=True,
        )

        eval_dataset = TCGA_BRCA_BagDataset(
            h5_dir=config.h5_dir,
            slides=train_val_slides,
            is_train=False,
        )

        test_dataset = TCGA_BRCA_BagDataset(
            h5_dir=config.h5_dir,
            slides=test_slides,
            is_train=False,
        )

        # Summarize class distributions after dataset filtering.
        train_dist_after = dict(Counter(s["target"] for s in train_dataset.slides))
        test_dist_after  = dict(Counter(s["target"] for s in test_dataset.slides))

        logger.info("=" * 60)
        logger.info(
            "Global split: Total=%d | Train+Val=%d | Test=%d",
            total_global,
            len(train_val_slides),
            len(test_slides),
        )
        logger.info("=" * 60)
        logger.info("Class distribution before and after filtering")
        logger.info(
            "Train+Val: %s -> %s (%d slides)",
            train_dist_before,
            train_dist_after,
            len(train_dataset),
        )
        logger.info(
            "Test: %s -> %s (%d slides)",
            test_dist_before,
            test_dist_after,
            len(test_dataset),
        )
        logger.info("=" * 60)
        logger.info("Datasets initialized successfully")

    except Exception:
        logger.exception("Dataset initialization failed")
        sys.exit(1)

    # Select the computation device.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Compute device: %s", device)

    # Run cross-validation, final training, and test evaluation.
    run_stratified_kfold_training(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        model_class=AttentionMIL,
        device=device,
        config=config,
    )

    logger.info("Training completed")


if __name__ == "__main__":
    main()
