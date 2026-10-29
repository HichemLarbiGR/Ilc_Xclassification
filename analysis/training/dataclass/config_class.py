
"""Configuration loading and validation for MIL training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class TrainingConfig:
    """Store the configuration for model training and evaluation."""

    # Data and output paths.
    h5_dir: str
    path_save_results: str
    save_model_path: str
    json_split_wsi_target_path: str

    # Model and feature parameters.
    input_dim: int
    num_classes: int
    model_name: str
    scale_factor: int
    scale: float = field(init=False)

    # Training hyperparameters.
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay_kfold: float
    weight_decay_final: float
    label_smoothing: float
    patience_stop: int
    factor_sche: 0.5
    patience_sche: 10
    min_lr_sche: 1e-6

    # Evaluation parameters.
    n_splits: int
    test_size: float
    monitor_test_size: float

    def __post_init__(self) -> None:
        """Validate parameters and compute derived values."""
        if self.scale_factor <= 0:
            raise ValueError("scale_factor must be greater than zero.")

        if self.input_dim <= 0:
            raise ValueError("input_dim must be greater than zero.")

        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        if self.epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")

        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2.")

        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1.")

        if not 0.0 < self.monitor_test_size < 1.0:
            raise ValueError(
                "monitor_test_size must be between 0 and 1."
            )

        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(
                "label_smoothing must be in the interval [0, 1)."
            )

        self.scale = 1.0 / self.scale_factor
        
    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainingConfig":
        """Load a training configuration from a YAML file."""
        yaml_path = Path(config_path).expanduser().resolve()
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

        root = yaml_path.parent.parent

        def resolve(value: str) -> str:
            path = Path(value)
            return str(path if path.is_absolute() else root / path)

        return cls(
            h5_dir=resolve(raw["paths"]["h5_dir"]),
            path_save_results=resolve(raw["paths"]["results"]),
            save_model_path=resolve(raw["paths"]["model_weights"]),
            json_split_wsi_target_path=resolve(raw["paths"]["json_split_wsi_target_path"]),
            model_name=raw["model"]["name"],
            input_dim=raw["model"]["input_dim"],
            num_classes=raw["model"]["num_classes"],
            scale_factor=raw["model"]["scale_factor"],
            epochs=raw["training"]["epochs"],
            batch_size=raw["training"]["batch_size"],
            learning_rate=raw["training"]["learning_rate"],
            weight_decay_kfold=raw["training"]["weight_decay_kfold"],
            weight_decay_final=raw["training"]["weight_decay_final"],
            label_smoothing=raw["training"]["label_smoothing"],
            patience_stop=raw["training"]["patience_stop"],
            factor_sche= raw["training"]["factor_sche"],
            patience_sche= raw["training"]["patience_sche"],
            min_lr_sche= raw["training"]["min_lr_sche"],
            n_splits=raw["evaluation"]["n_splits"],
            test_size=raw["evaluation"]["test_size"],
            monitor_test_size=raw["evaluation"]["monitor_test_size"],
        )