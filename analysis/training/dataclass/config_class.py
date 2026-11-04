
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
    png_path: str
    wsi_path: str
    seed: int

    # Model and feature parameters.
    input_dim: int
    num_classes: int
    classes_name: str
    architecture_name: str
    foundation_model_name: str
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
    factor_sche: float
    patience_sche: int
    min_lr_sche: float

    # Evaluation parameters.
    n_splits: int
    test_size: float
    monitor_test_size: float
    bootstrap_niter: int
    n_tta: int
    infer_model: str
    subset_name: str
    figures_extension: str

    #XAI
    scale_factor: int
    top_k: int
    bottom_k: int
    score_threshold: int
    cmap: str
    alpha: float
    xai_model: str
    top_k_perc: int
    class_weights: tuple[float, ...] = (1.0, 4.0)

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
            path_save_results=resolve(raw["paths"]["path_results"]),
            save_model_path=resolve(raw["paths"]["save_model_path"]),
            json_split_wsi_target_path=resolve(raw["paths"]["json_split_wsi_target_path"]),
            png_path=raw["paths"]["png_path"],
            wsi_path=raw["paths"]["wsi_path"],
            seed=raw["paths"]["seed"],
            class_weights=raw["model"]["class_weights"],
            architecture_name=raw["model"]["architecture_name"],
            foundation_model_name=raw["model"]["foundation_model_name"],
            input_dim=raw["model"]["input_dim"],
            num_classes=raw["model"]["num_classes"],
            classes_name=raw["model"]["classes_name"],
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
            bootstrap_niter=raw["evaluation"]["bootstrap_niter"],
            n_tta=raw["evaluation"]["n_tta"],
            infer_model=raw["evaluation"]["infer_model"],
            subset_name=raw["evaluation"]["subset_name"],
            figures_extension=raw["evaluation"]["figures_extension"],
            scale_factor=raw["xai"]["scale_factor"],
            top_k=raw["xai"]["top_k"],
            bottom_k=raw["xai"]["bottom_k"],
            score_threshold=raw["xai"]["score_threshold"],
            cmap=raw["xai"]["cmap"],
            alpha=raw["xai"]["alpha"]
            xai_model=raw["xai"]["xai_model"]
            top_k_perc=raw["xai"]["top_k_perc"]
        )