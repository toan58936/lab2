"""
Training engine: train_step, evaluate_step, train loop.

Phase 5: canonical API with train_mode_fn, 6-key history,
checkpoint resume, atomic persistence via utils.py, and TensorBoard
tag/step conventions per docs/define&designt/03_engine.md.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn, optim

from going_modular import utils


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def train_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device | str,
    train_mode_fn: Optional[Callable[[nn.Module], None]] = None,
) -> Tuple[float, float]:
    """One training epoch. Returns (sample-weighted loss, accuracy).

    If ``train_mode_fn`` is supplied it is called once at the start to put
    the model in train mode (e.g. ``functools.partial(set_train_mode, cfg=...)``
    which keeps frozen BatchNorm in eval). Otherwise ``model.train()`` is used.
    """
    if train_mode_fn is not None:
        train_mode_fn(model)
    else:
        model.train()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training dataloader produced no samples.")
    return total_loss / total_samples, total_correct / total_samples


def evaluate_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device | str,
) -> Tuple[float, float]:
    """One evaluation epoch (no_grad). Returns (sample-weighted loss, accuracy).

    Leaves the model in eval mode.
    """
    model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            batch_size = y.size(0)
            total_loss += loss_fn(logits, y).item() * batch_size
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Evaluation dataloader produced no samples.")
    return total_loss / total_samples, total_correct / total_samples


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device | str,
    train_mode_fn: Optional[Callable[[nn.Module], None]] = None,
    writer=None,
    start_epoch: int = 0,
    save_path: Optional[str] = None,
    monitor: str = "val_acc",
    restore_best: bool = True,
    checkpoint_dir: Optional[str] = None,
    resume: bool = False,
    expected_phase_config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, List]:
    """Multi-epoch training loop with best-checkpoint selection and resume.

    History keys: train_loss, val_loss, train_acc, val_acc, lr, epoch_time.
    Each key holds one entry per completed epoch.
    """
    # ---- Validate inputs --------------------------------------------------
    if epochs <= 0:
        raise ValueError(f"epochs must be positive; got {epochs}.")
    if start_epoch < 0:
        raise ValueError(f"start_epoch must be >= 0; got {start_epoch}.")
    if monitor not in ("val_acc", "val_loss"):
        raise ValueError(
            f"monitor must be 'val_acc' or 'val_loss'; got '{monitor}'."
        )

    device_t = device if isinstance(device, torch.device) else torch.device(device)
    model.to(device_t)

    # ---- Resume or fresh start -------------------------------------------
    phase_epochs_completed = 0
    history: Dict[str, List] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "lr": [], "epoch_time": [],
    }

    if resume:
        if checkpoint_dir is None:
            raise ValueError("resume=True requires checkpoint_dir.")
        ckpt_path = Path(checkpoint_dir) / "last_checkpoint.pth"
        phase_epochs_completed, best_score, best_epoch, history = utils.load_checkpoint(
            str(ckpt_path),
            model,
            optimizer,
            device_t,
            expected_phase_config or {},
            arch=(expected_phase_config or {}).get("arch"),
        )
        if phase_epochs_completed >= epochs:
            raise ValueError(
                f"Checkpoint already completed {phase_epochs_completed} epochs "
                f">= requested {epochs}. Nothing to resume."
            )
    else:
        best_score: float | None = None
        best_epoch: int | None = None

    # ---- Determine monitor direction --------------------------------------
    maximize = monitor == "val_acc"
    if best_score is None:
        best_score = float("-inf") if maximize else float("inf")
    if best_epoch is None:
        best_epoch = None

    # ---- Training loop -----------------------------------------------------
    for phase_epoch in range(phase_epochs_completed, epochs):
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_step(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            device_t,
            train_mode_fn=train_mode_fn,
        )
        val_loss, val_acc = evaluate_step(
            model, val_dataloader, loss_fn, device_t
        )

        if device_t.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.perf_counter() - epoch_start

        # LR per param group
        lr_values = [group["lr"] for group in optimizer.param_groups]

        # Append history (one item per epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_values)
        history["epoch_time"].append(epoch_time)

        global_step = start_epoch + phase_epoch + 1

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, global_step)
            writer.add_scalar("Loss/val", val_loss, global_step)
            writer.add_scalar("Accuracy/train", train_acc, global_step)
            writer.add_scalar("Accuracy/val", val_acc, global_step)
            for i, lr in enumerate(lr_values):
                writer.add_scalar(f"LR/group_{i}", lr, global_step)

        # Monitor improvement (strict comparison; ties keep earlier epoch)
        current = val_acc if maximize else val_loss
        if maximize:
            improved = current > best_score
        else:
            improved = current < best_score

        if improved:
            best_score = current
            best_epoch = phase_epoch
            if save_path is not None:
                save_p = Path(save_path)
                save_p.parent.mkdir(parents=True, exist_ok=True)
                utils.save_model(model, str(save_p))

        # Checkpoint + history persistence (atomic via utils)
        if checkpoint_dir is not None:
            ckpt_dir = Path(checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            phase_config = expected_phase_config or {}
            utils.save_checkpoint(
                str(ckpt_dir / "last_checkpoint.pth"),
                model,
                optimizer,
                phase_epochs_completed=phase_epoch + 1,
                best_score=best_score,
                best_epoch=best_epoch if best_epoch is not None else phase_epoch,
                history=history,
                phase_config=phase_config,
            )
            utils.save_history(str(ckpt_dir), history)

        if verbose:
            print(
                f"Epoch: {phase_epoch + 1} | "
                f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
            )

    # ---- Finalize ----------------------------------------------------------
    if checkpoint_dir is not None:
        utils.save_history(str(Path(checkpoint_dir)), history)

    if restore_best and save_path is not None and Path(save_path).is_file():
        model.load_state_dict(torch.load(save_path, map_location=device_t))

    return history
