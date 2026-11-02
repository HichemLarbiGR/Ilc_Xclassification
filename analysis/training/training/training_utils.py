"""Training, validation, cross-validation, and evaluation utilities.

This module provides the training and evaluation routines used for
multiple-instance learning (MIL):

* single-epoch training and evaluation,
* per-fold training with early stopping and best-checkpoint restoration,
* final model training on the full train+validation set,
* test-time augmentation (TTA) evaluation,
* stratified K-fold cross-validation orchestration, with bootstrap
  confidence intervals on the test set and JSON/checkpoint export.
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
from utils.utils import logger, save_run_json, build_name_model_pth


def _extract_logits(model_output: Any) -> torch.Tensor:
    """Extract the logits from a model output.

    Supports models returning either a tuple/list whose first element is
    the logits (the remaining elements being auxiliary outputs), or a
    tensor containing the logits directly.

    Args:
        model_output: Raw output of the model's forward pass.

    Returns:
        The logits tensor.
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
) -> Tuple[float, Dict, None]:
    """Train the model for one epoch.

    The optimisation step uses the batch-mean loss. The returned loss is
    the epoch loss obtained by weighting each batch-mean loss by its
    batch size (consistent with `evaluate_model`). Batches that are
    entirely filtered out, as well as batches yielding a non-finite loss,
    are skipped and do not contribute to the returned loss or metrics.

    Args:
        model: Model to train.
        dataloader: Training data loader.
        optimizer: Optimizer used to update the model parameters.
        device: Device on which computations are performed.
        config: Training configuration.

    Returns:
        A tuple containing the epoch loss, the overall metrics, and
        ``None`` (placeholder for per-class metrics, not computed).

    Raises:
        ValueError: If no sample was processed during the epoch.
    """
    model.train()
    class_weights = torch.tensor(config.class_weights, dtype=torch.float32).to(device)
    # Class-weighted cross-entropy with label smoothing (mean reduction).
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.label_smoothing,
        weight=class_weights
    )

    total_loss = 0.0
    n_samples = 0
    n_skipped_batches = 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in dataloader:
        if batch is None:
            # The collate function returns None when all the slides of
            # the batch were filtered out.
            n_skipped_batches += 1
            continue

        bags, labels, _ = batch
        labels = labels.to(device)
        bags = [bag.to(device) for bag in bags]

        logits = _extract_logits(model(bags))
        loss = criterion(logits, labels)  # Batch-mean loss.

        # Never backpropagate a non-finite loss: it would corrupt the weights.
        if not torch.isfinite(loss):
            logger.warning("train_one_epoch: non-finite loss, batch skipped.")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        # Weight the batch-mean loss by the batch size to accumulate the
        # epoch loss. This is an exact per-sample mean without class
        # weights; with class weights, PyTorch normalises each batch mean
        # by the sum of the sample weights, so it is an approximation.
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
) -> Tuple[float, Dict, None, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the model on a validation or test dataset.

    The model is put in evaluation mode and no gradient is computed.
    The loss uses the same criterion as `train_one_epoch`.

    Args:
        model: Model to evaluate.
        dataloader: Evaluation data loader.
        device: Device on which computations are performed.
        config: Training configuration.

    Returns:
        A tuple containing the epoch loss (batch-mean losses weighted by
        batch size), the overall metrics, ``None`` (placeholder for
        per-class metrics, not computed), the true labels, the predicted
        labels, and the predicted class probabilities.
    """
    model.eval()

    total_loss = 0.0
    n_samples = 0
    all_preds, all_labels, all_probs = [], [], []
    class_weights = torch.tensor(config.class_weights, dtype=torch.float32).to(device)
    # Class-weighted cross-entropy with label smoothing (mean reduction).
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.label_smoothing,
        weight=class_weights
    )

    with torch.no_grad():
        for bags, labels, _ in dataloader:
            labels = labels.to(device)
            bags = [bag.to(device) for bag in bags]

            logits = _extract_logits(model(bags))
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)

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
        total_loss / n_samples,
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
) -> Tuple[List[float], List[float], Dict, Dict, int, Optional[Dict[str, Any]]]:
    """Train one cross-validation fold with early stopping.

    The model state corresponding to the lowest validation loss is
    restored before the final training and validation metrics are
    computed, so the returned metrics reflect the best checkpoint rather
    than the last epoch. Epochs with a non-finite loss are ignored, and
    training is aborted after three consecutive such epochs.

    Args:
        model: Model to train (moved to `device` by this function).
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Device on which computations are performed.
        config: Training configuration.

    Returns:
        A tuple containing:

        * the per-epoch training losses,
        * the per-epoch validation losses,
        * the final training metrics (best checkpoint, evaluation mode),
        * the final validation metrics (best checkpoint),
        * the number of epochs actually run,
        * the best state dict (a deep copy, kept on the model's device),
          or ``None`` if no valid checkpoint was saved.
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
    losses_train_epochs_list = []
    losses_val_epochs_list = []
    results_folds = {}

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

        losses_train_epochs_list.append(train_loss)
        losses_val_epochs_list.append(val_loss)

        # Skip non-finite epochs: no scheduler step, no early-stopping
        # update, no checkpoint. Abort if the model has clearly diverged.
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
            "No valid checkpoint was saved: "
            "returning the model from the last epoch."
        )

    # Final metrics computed in evaluation mode with the restored weights.
    _, final_train_metrics, _, _, _, _ = evaluate_model(
        model, train_loader, device, config
    )
    _, final_val_metrics, _, _, _, _ = evaluate_model(
        model, val_loader, device, config
    )

    return (
        losses_train_epochs_list,
        losses_val_epochs_list,
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
) -> Tuple[Dict[str, Any], nn.Module]:
    """Train the final model on the train+validation data.

    A stratified monitoring subset is held out from `train_val_idx`; it is
    used for learning-rate scheduling, early stopping, and checkpoint
    selection, and is therefore not used for gradient updates. The best
    model state (lowest monitoring loss) is restored before returning.

    Args:
        model_class: Model class, instantiated with ``num_classes`` and
            ``in_dim`` taken from the configuration.
        train_dataset: Dataset used for training (augmented).
        eval_dataset: Dataset used for monitoring (non-augmented). It must
            contain the same slides, in the same order, as `train_dataset`.
        train_val_idx: Indices of the slides used for training and
            monitoring.
        device: Device on which computations are performed.
        config: Training configuration.

    Returns:
        A tuple containing a results dictionary and the trained model in
        evaluation mode. The dictionary holds the loss and metrics of the
        last epoch run, for both the training and the monitoring sets
        (not those of the restored best checkpoint).

    Raises:
        AssertionError: If `train_dataset` and `eval_dataset` are not
            aligned.
    """
    losses_final_model_train_list = []
    losses_final_model_monitor_list = []

    # Check that both datasets contain the same slides in the same order.
    train_ids = [s["id"] for s in train_dataset.slides]
    eval_ids = [s["id"] for s in eval_dataset.slides]
    assert train_ids == eval_ids, (
        "train_dataset and eval_dataset are not aligned "
        "(slides differ in content or order)."
    )

    # Labels are read directly from the dataset metadata (no H5 I/O).
    train_val_idx = list(train_val_idx)
    labels = np.array(
        [train_dataset.slides[i]["target"] for i in train_val_idx]
    )

    train_idx, monitor_idx = train_test_split(
        train_val_idx,
        test_size=config.monitor_test_size,
        stratify=labels,
        random_state=config.seed,
    )

    # Reproducible batch order.
    generator = torch.Generator().manual_seed(config.seed)

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=config.batch_size,
        collate_fn=collate_bags,
        shuffle=True,
        generator=generator,
    )

    # The monitoring indices refer to the same slide positions in both
    # (aligned) datasets, so they can be applied to `eval_dataset`.
    monitor_loader = DataLoader(
        Subset(eval_dataset, monitor_idx),
        batch_size=config.batch_size,
        collate_fn=collate_bags,
        shuffle=False,
    )

    # Deterministic model initialisation.
    torch.manual_seed(config.seed)
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
        final_model_train_loss, final_model_train_metrics, _ = train_one_epoch(
            model, train_loader, optimizer, device, config
        )

        monitor_loss, monitor_metrics, _, _, _, _ = evaluate_model(
            model, monitor_loader, device, config
        )
        losses_final_model_train_list.append(final_model_train_loss)
        losses_final_model_monitor_list.append(monitor_loss)

        # Skip non-finite epochs: no scheduler step, no early-stopping
        # update, no checkpoint. A NaN is never considered an improvement.
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
            # Deep copy of the weights (kept on the model's device).
            best_state = copy.deepcopy(model.state_dict())

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

    # Loss and metrics of the last epoch run.
    results_final_model = {
        "train_final_model": {
            "losses": final_model_train_loss,
            "metrics": final_model_train_metrics
        },
        "monitor_final_model": {
            "losses": monitor_loss,
            "metrics": monitor_metrics
        }
    }

    model.eval()
    return results_final_model, model


def evaluate_model_tta(
    model: nn.Module,
    dataset,
    device: torch.device,
    config: "TrainingConfig",
) -> Tuple[
    float,
    Dict,
    None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List
]:
    """Evaluate a model using test-time augmentation (TTA).

    For each slide, the class probabilities are averaged over
    ``config.n_tta`` random patch samplings (obtained by temporarily
    enabling the dataset's training mode). The dataset's original
    training mode is restored afterwards, even if an error occurs.

    Args:
        model: Model to evaluate.
        dataset: Dataset to evaluate, exposing an ``is_train`` attribute.
        device: Device on which computations are performed.
        config: Training configuration (uses ``n_tta`` and
            ``num_classes``).

    Returns:
        A tuple containing a placeholder loss (always ``0.0``, no loss is
        computed), the overall metrics, ``None`` (placeholder for
        per-class metrics), the true labels, the predicted labels, the
        averaged class probabilities, and the slide identifiers.

    Raises:
        ValueError: If ``config.n_tta`` is lower than 1.
    """
    n_draws = config.n_tta

    if n_draws < 1:
        raise ValueError("config.n_tta must be at least 1.")

    model.eval()

    original_is_train = dataset.is_train

    all_preds, all_labels, all_probs, all_ids = [], [], [], []

    try:
        # Enable random patch sampling to obtain different draws per slide.
        dataset.is_train = True

        with torch.no_grad():
            for idx in range(len(dataset)):
                # First access: only used to retrieve the label and the
                # slide identifier (the sampled bag is discarded).
                _, label, case_id = dataset[idx]
                label_value = (
                    label.item() if torch.is_tensor(label) else label
                )

                all_ids.append(case_id)

                probs_draws = []

                for _ in range(n_draws):
                    bags, _, _ = dataset[idx]
                    bags = [bags.to(device)]

                    logits = _extract_logits(model(bags))
                    probs = torch.softmax(logits, dim=1)

                    probs_draws.append(probs.cpu().numpy())

                # Average the probabilities over the draws, then predict.
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
        all_ids
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

    The pipeline consists of:

    1. a stratified K-fold cross-validation on the train+validation set,
    2. the training of a final model on the whole train+validation set,
    3. the evaluation of this model on the test set with TTA,
    4. the computation of 95% bootstrap confidence intervals,
    5. the export of the results (JSON) and of the final model weights.

    Args:
        train_dataset: Training dataset (augmented), used with
            `eval_dataset` for the train+validation data.
        eval_dataset: Evaluation dataset (non-augmented), aligned with
            `train_dataset`.
        test_dataset: Held-out test dataset.
        model_class: Model class to instantiate for each training run.
        device: Device on which computations are performed.
        config: Training configuration.

    Returns:
        A dictionary containing the configuration, the per-fold results
        (losses, metrics, number of epochs), and the test metrics with
        their bootstrap confidence intervals. The final model weights are
        saved to disk rather than returned.
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
        random_state=config.seed,
    )
    train_losses_list = []
    train_metrics_list = []
    val_metrics_list = []
    best_states = []
    final_epochs = []
    results_folds = {}

    # Cross-validation.
    for fold, (train_local, val_local) in enumerate(
        skf.split(train_val_idx, labels),
        start=1
    ):
        logger.info(
            f"\n{'=' * 60}\n"
            f"Fold {fold}/{config.n_splits}\n"
            f"{'=' * 60}"
        )
        fold_name = "Fold " + str(fold)
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

        train_losses_list, val_losses_list, train_metrics, val_metrics, final_epoch, best_state = train_fold(
            model, train_loader, val_loader, device, config
        )

        logger.info(
            f"Fold {fold} results | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        results_folds[fold_name] = {
            "Train losses": train_losses_list,
            "Val losses": val_losses_list,
            "Train metrics": train_metrics,
            "Val metrics": val_metrics,
            "Final epoch": final_epoch
        }
        train_metrics_list.append(train_metrics)
        val_metrics_list.append(val_metrics)
        best_states.append(best_state)
        final_epochs.append(final_epoch)

    # Train the final model on the train+validation data.
    logger.info("\nTraining the final model on Train+Val...")

    results_final_model, final_model = train_final_model(
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
        f"({config.n_tta} draws)..."
    )

    (
        _,
        test_metrics,
        test_per_class,
        test_labels,
        test_preds,
        test_probs,
        test_ids
    ) = evaluate_model_tta(
        final_model,
        test_dataset,
        device,
        config,
    )

    logger.info(
        f"\nTest set | Accuracy: {test_metrics['accuracy']:.4f} | "
        f"F1-macro: {test_metrics['f1_macro']:.4f}"
    )

    # Compute the bootstrap confidence intervals.
    logger.info("\n" + "=" * 50 + "\n95% Bootstrap CIs\n" + "=" * 50)

    ci_metrics = {}

    for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "auc"):
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

    # Assemble the results to export (configuration, folds, test).
    cfg = {"Configuration": {k: v for k, v in vars(config).items()}}

    results_test = {
        "Test": ci_metrics
    }

    results_json = {
        **cfg,
        **results_folds,
        **results_test
    }

    # Save the results and the final model weights.
    save_directory = config.path_save_results

    if save_directory:
        os.makedirs(save_directory, exist_ok=True)

    run_id, time_save, save_results_path = save_run_json(results_json, save_dir=save_directory)

    model_name_path = build_name_model_pth(config.save_model_path, config.architecture_name, config.foundation_model_name, run_id)
    
    os.makedirs(os.path.dirname(model_name_path), exist_ok=True)
    
    torch.save(final_model.state_dict(), model_name_path)

    logger.info(f"Final model saved to: {model_name_path}")

    return results_json