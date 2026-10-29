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
normalize_slide_labels
    Convert raw slide labels to integer case IDs and labels.
"""

import json
import logging
import math
from dataclasses import asdict
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)

SEED = 42

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


def normalize_slide_labels(raw_labels):
    """Convert valid slide labels to integer case IDs and labels.

    Labels may be stored as integers or as lists whose first element
    contains the label. Missing values and scalar NaN values are skipped.
    """
    normalized_labels = {}

    for key, value in raw_labels.items():
        if value is None:
            continue

        if isinstance(value, float) and math.isnan(value):
            continue

        label = value[0] if isinstance(value, list) else value
        normalized_labels[int(key)] = int(label)

    return normalized_labels