"""Training, validation, cross-validation, and evaluation utilities.

This module provides the training and evaluation routines used for
multiple-instance learning (MIL), including stratified K-fold
cross-validation, final model training, test-time augmentation (TTA),
and bootstrap confidence intervals.
"""

"""Training, validation, cross-validation, and evaluation utilities.

This module provides the training and evaluation routines used for
multiple-instance learning (MIL), including stratified K-fold
cross-validation, final model training, test-time augmentation (TTA),
and bootstrap confidence intervals.
"""

"""Training, validation, cross-validation, and evaluation utilities.

This module provides the training and evaluation routines used for
multiple-instance learning (MIL), including stratified K-fold
cross-validation, final model training, test-time augmentation (TTA),
and bootstrap confidence intervals.
"""

import copy
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset

from dataclass.config_class import TrainingConfig
from dataloader.data_loader_bag import collate_bags
from models.MIL_models import EarlyStopping
from training.bootstrap import bootstrap_ci
from training.metrics import get_metrics
from utils.utils import SEED, logger

SEED = 42 

def _extract_logits(model_output: Any) -> torch.Tensor:
    """Extract logits from a model output.

    Supports models returning either a tuple containing logits and
    auxiliary outputs or a tensor containing logits directly.
    """
    if isinstance(model_output, (tuple, list)):
        return model_output[0]

    return model_output


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: "TrainingConfig",
) -> Tuple[float, Dict, Dict]:
    """Train the model for one epoch.

    The optimisation uses the batch-mean loss, while the returned loss is
    the mean per-sample loss over the epoch (matching `evaluate_model`).

    Args:
        model: Model to train.
        dataloader: Training data loader.
        optimizer: Optimizer used to update model parameters.
        device: Device on which computations are performed.
        config: Training configuration.

    Returns:
        A tuple containing the mean per-sample loss, overall metrics,
        and per-class metrics.
    """
    model.train()

    # Mean reduction for the gradient step.
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.label_smoothing,
    )

    total_loss = 0.0
    n_samples = 0
    n_skipped_batches = 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in dataloader:
        if batch is None:
            # All slides in this batch were filtered out.
            n_skipped_batches += 1
            continue

        bags, labels, _ = batch
        labels = labels.to(device)
        bags = [bag.to(device) for bag in bags]

        logits = _extract_logits(model(bags))
        loss = criterion(logits, labels)  # batch mean

        # Never backpropagate a non-finite loss: it would poison the weights.
        if not torch.isfinite(loss):
            logger.warning("train_one_epoch: non-finite loss, batch skipped.")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        # Re-weight the batch mean by its size to get an exact epoch mean.
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        all_probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())

    if n_skipped_batches > 0:
        logger.debug(
            f"train_one_epoch: {n_skipped_batches} batch(es) fully "
            "filtered out and ignored."
        )

    if n_samples == 0:
        raise ValueError(
            "train_one_epoch received an empty or fully filtered dataloader."
        )

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    metrics = get_metrics(
        all_labels, all_preds, all_probs, config.num_classes
    )


    return total_loss / n_samples, metrics, None


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: "TrainingConfig",
) -> Tuple[float, Dict, Dict, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the model on a validation or test dataset.

    Args:
        model: Model to evaluate.
        dataloader: Evaluation data loader.
        device: Device on which computations are performed.
        config: Training configuration.

    Returns:
        A tuple containing the mean batch loss, overall metrics,
        per-class metrics, true labels, predictions, and probabilities.
    """
    model.eval()

    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.label_smoothing
    )

    with torch.no_grad():
        for bags, labels, _ in dataloader:
            labels = labels.to(device)
            bags = [bag.to(device) for bag in bags]

            logits = _extract_logits(model(bags))
            loss = criterion(logits, labels)

            total_loss += loss.item()

            all_preds.extend(
                torch.argmax(logits, dim=1).cpu().numpy()
            )
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(
                torch.softmax(logits, dim=1).cpu().numpy()
            )

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    metrics = get_metrics(
        all_labels, all_preds, all_probs, config.num_classes
    )

    return (
        total_loss / len(dataloader),
        metrics,
        None,
        all_labels,
        all_preds,
        all_probs,
    )


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: "TrainingConfig",
) -> Tuple[Dict, Dict, int, Optional[Dict[str, Any]]]:
    """Train one cross-validation fold with early stopping.

    The model state corresponding to the lowest validation loss is
    restored before the final training and validation metrics are
    computed, so the returned metrics reflect the best checkpoint,
    not the last epoch.

    Returns the final train metrics, final validation metrics, the number
    of epochs actually run, and the best state dict (on CPU).
    """
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay_kfold,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.factor_sche,
        patience=config.patience_sche,
        min_lr=config.min_lr_sche,
    )

    if config.patience_stop <= config.patience_sche:
        logger.warning(
            f"patience_stop ({config.patience_stop}) <= patience_sche "
            f"({config.patience_sche}): early stopping may trigger before "
            "the scheduler has reduced the learning rate."
        )

    early_stopping = EarlyStopping(patience=config.patience_stop)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    final_epoch = 0
    consecutive_bad_epochs = 0
    max_bad_epochs = 3

    for epoch in range(config.epochs):
        final_epoch = epoch + 1

        # Safety net: evaluate_model leaves the model in eval() mode.
        model.train()

        train_loss, train_metrics, _ = train_one_epoch(
            model, train_loader, optimizer, device, config
        )

        val_loss, val_metrics, _, _, _, _ = evaluate_model(
            model, val_loader, device, config
        )

        # Skip non-finite epochs: no scheduler step, no early stopping,
        # no checkpoint. Abort if the model has clearly diverged.
        if not (np.isfinite(train_loss) and np.isfinite(val_loss)):
            consecutive_bad_epochs += 1
            logger.warning(
                f"Epoch {epoch + 1}: non-finite loss "
                f"(train={train_loss}, val={val_loss}), epoch ignored "
                f"({consecutive_bad_epochs}/{max_bad_epochs})."
            )
            if consecutive_bad_epochs >= max_bad_epochs:
                logger.error("Training diverged, stopping this fold.")
                break
            continue
        consecutive_bad_epochs = 0

        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} "
            f"Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_loss:.4f} "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
      
            logger.info(f"New best validation loss: {best_val_loss:.4f}")

        early_stopping(val_loss)

        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(
            f"Best checkpoint restored (epoch {best_epoch}, "
            f"val loss {best_val_loss:.4f})."
        )
    else:
        logger.warning(
            "No valid checkpoint was saved (all losses non-finite?): "
            "returning the model from the last epoch."
        )

    _, final_train_metrics, _, _, _, _ = evaluate_model(
        model, train_loader, device, config
    )
    _, final_val_metrics, _, _, _, _ = evaluate_model(
        model, val_loader, device, config
    )

    return (
        final_train_metrics,
        final_val_metrics,
        final_epoch,
        best_state,
    )


def train_final_model(
    model_class: type,
    train_dataset,
    eval_dataset,
    train_val_idx: List[int],
    device: torch.device,
    config: "TrainingConfig",
) -> nn.Module:
    """Train the final model using the training and validation data.

    A stratified monitoring subset is used for learning-rate scheduling
    and early stopping. The best model state is restored before return.
    """
    # Alignement réel : mêmes slides dans le même ordre.
    train_ids = [s["id"] for s in train_dataset.slides]
    eval_ids = [s["id"] for s in eval_dataset.slides]
    assert train_ids == eval_ids, (
        "train_dataset and eval_dataset are not aligned "
        "(slides differ in content or order)."
    )

    # Metadata read directly from the split, no H5 I/O.
    train_val_idx = list(train_val_idx)
    labels = np.array(
        [train_dataset.slides[i]["target"] for i in train_val_idx]
    )

    train_idx, monitor_idx = train_test_split(
        train_val_idx,
        test_size=config.monitor_test_size,
        stratify=labels,
        random_state=SEED,
    )

    # Reproducible batch order.
    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=config.batch_size,
        collate_fn=collate_bags,
        shuffle=True,
        generator=generator,
    )

    # monitor_idx indexes train_val_idx, valid for both aligned datasets.
    monitor_loader = DataLoader(
        Subset(eval_dataset, monitor_idx),
        batch_size=config.batch_size,
        collate_fn=collate_bags,
        shuffle=False,
    )

    # Deterministic model initialization.
    torch.manual_seed(SEED)
    model = model_class(
        num_classes=config.num_classes,
        in_dim=config.input_dim,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay_final,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.factor_sche,
        patience=config.patience_sche,
        min_lr=config.min_lr_sche,
    )

    if config.patience_stop <= config.patience_sche:
        logger.warning(
            f"patience_stop ({config.patience_stop}) <= patience "
            f"({config.patience_sche}): early stopping may trigger before "
            "the scheduler reduces the learning rate."
        )

    early_stopping = EarlyStopping(patience=config.patience_stop)

    best_monitor_loss = float("inf")
    best_state = None

    for epoch in range(config.epochs):
        # Safety net: evaluate_model leaves the model in eval() mode.
        model.train()
        train_one_epoch(
            model, train_loader, optimizer, device, config
        )

        monitor_loss, monitor_metrics, _, _, _, _ = evaluate_model(
            model, monitor_loader, device, config
        )

        # Skip non-finite epochs: no scheduler, no early stopping,
        # no checkpoint. NaN does not count as an improvement.
        if not np.isfinite(monitor_loss):
            logger.warning(
                f"Epoch {epoch + 1}: non-finite monitor loss "
                f"({monitor_loss}), epoch ignored."
            )
            continue

        scheduler.step(monitor_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Final Training Epoch {epoch + 1}/{config.epochs} | "
                f"Monitor Loss: {monitor_loss:.4f} "
                f"Acc: {monitor_metrics['accuracy']:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        if monitor_loss < best_monitor_loss:
            best_monitor_loss = monitor_loss
            # Copy to CPU to avoid GPU memory saturation.
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        early_stopping(monitor_loss)

        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(
            "Final model restored with best monitoring loss: "
            f"{best_monitor_loss:.4f}"
        )
    else:
        logger.warning(
            "No best state saved (non-finite loss at every epoch?): "
            "returning the model from the last epoch."
        )

    model.eval()
    return model


def evaluate_model_tta(
    model: nn.Module,
    dataset,
    device: torch.device,
    config: "TrainingConfig",
    n_draws: int = 5,
) -> Tuple[
    float,
    Dict,
    Dict,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Evaluate a model using test-time augmentation.

    Predictions are averaged across multiple random patch samples for
    each slide. The dataset's original training mode is restored afterward.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1.")

    model.eval()

    original_is_train = dataset.is_train

    all_preds, all_labels, all_probs = [], [], []

    try:
        dataset.is_train = True

        with torch.no_grad():
            for idx in range(len(dataset)):
                _, label, _ = dataset[idx]
                label_value = (
                    label.item() if torch.is_tensor(label) else label
                )

                probs_draws = []

                for _ in range(n_draws):
                    bags, _, _ = dataset[idx]
                    bags = [bags.to(device)]

                    logits = _extract_logits(model(bags))
                    probs = torch.softmax(logits, dim=1)

                    probs_draws.append(probs.cpu().numpy())

                mean_probs = np.mean(probs_draws, axis=0)
                pred = np.argmax(mean_probs, axis=1)[0]

                all_labels.append(label_value)
                all_preds.append(pred)
                all_probs.append(mean_probs[0])

    finally:
        dataset.is_train = original_is_train

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    metrics = get_metrics(
        all_labels, all_preds, all_probs, config.num_classes
    )


    return (
        0.0,
        metrics,
        None,
        all_labels,
        all_preds,
        all_probs,
    )


def run_stratified_kfold_training(
    train_dataset,
    eval_dataset,
    test_dataset,
    model_class: type,
    device: torch.device,
    config: "TrainingConfig",
) -> Dict[str, Any]:
    """Run stratified K-fold training, final training, and test evaluation.

    Returns:
        A dictionary containing averaged training metrics, averaged
        validation metrics, test metrics, and the final model state.
    """
    

    train_val_idx = list(range(len(train_dataset)))

    labels = np.array([train_dataset.slides[i]["target"] for i in train_val_idx])

    logger.info(
        f"Data split: Train+Val K-Fold ({len(train_dataset)} slides) | "
        f"Test ({len(test_dataset)} slides)"
    )

    skf = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=SEED,
    )

    train_metrics_list = []
    val_metrics_list = []
    best_states = []
    final_epochs = []

    # Run cross-validation.
    for fold, (train_local, val_local) in enumerate(
        skf.split(train_val_idx, labels),
        start=1
    ):
        logger.info(
            f"\n{'=' * 60}\n"
            f"Fold {fold}/{config.n_splits}\n"
            f"{'=' * 60}"
        )

        train_indices = [train_val_idx[i] for i in train_local]
        val_indices = [train_val_idx[i] for i in val_local]

        train_loader = DataLoader(
            Subset(train_dataset, train_indices),
            batch_size=config.batch_size,
            collate_fn=collate_bags,
            shuffle=True,
        )

        val_loader = DataLoader(
            Subset(eval_dataset, val_indices),
            batch_size=config.batch_size,
            collate_fn=collate_bags,
            shuffle=False,
        )

        model = model_class(
            num_classes=config.num_classes,
            in_dim=config.input_dim,
        )

        train_metrics, val_metrics, final_epoch, best_state = train_fold(
            model, train_loader, val_loader, device, config
        )

        logger.info(
            f"Fold {fold} results | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        train_metrics_list.append(train_metrics)
        val_metrics_list.append(val_metrics)
        best_states.append(best_state)
        final_epochs.append(final_epoch)

    avg_train = {
        key: np.mean([metrics[key] for metrics in train_metrics_list])
        for key in train_metrics_list[0]
    }
    avg_val = {
        key: np.mean([metrics[key] for metrics in val_metrics_list])
        for key in val_metrics_list[0]
    }

    logger.info(
        f"\nCross-validation averages | "
        f"Train Acc: {avg_train['accuracy']:.4f} | "
        f"Val Acc: {avg_val['accuracy']:.4f}"
    )

    # Train the final model on the training and validation data.
    logger.info("\nTraining the final model on Train+Val...")
    final_model = train_final_model(
        model_class,
        train_dataset,
        eval_dataset,
        train_val_idx,
        device,
        config,
    )

    # Evaluate the final model on the test set using TTA.
    logger.info(
        "\nEvaluating the test set with test-time augmentation "
        "(5 draws)..."
    )

    (
        _,
        test_metrics,
        test_per_class,
        test_labels,
        test_preds,
        test_probs,
    ) = evaluate_model_tta(
        final_model,
        test_dataset,
        device,
        config,
        n_draws=5,
    )

    logger.info(
        f"\nTest set | Accuracy: {test_metrics['accuracy']:.4f} | "
        f"F1-macro: {test_metrics['f1_macro']:.4f}"
    )

    # Compute bootstrap confidence intervals.
    logger.info("\n" + "=" * 50 + "\n95% Bootstrap CIs\n" + "=" * 50)

    ci_metrics = {}

    for metric in ("accuracy", "f1_macro", "auc"):
        mean, low, high = bootstrap_ci(
            test_labels,
            test_probs,
            test_preds,
            metric,
        )

        ci_metrics[metric] = (mean, low, high)

        logger.info(
            f"{metric.capitalize():12} : {mean:.4f} "
            f"[CI: {low:.4f} - {high:.4f}]"
        )

    # Save the final model state.
    save_directory = os.path.dirname(config.save_model_path)

    if save_directory:
        os.makedirs(save_directory, exist_ok=True)

    torch.save(final_model.state_dict(), config.save_model_path)

    logger.info(f"Final model saved to: {config.save_model_path}")

    return {
        "train_metrics": avg_train,
        "val_metrics": avg_val,
        "test_metrics": test_metrics,
        "final_model_state": final_model.state_dict(),
    }

