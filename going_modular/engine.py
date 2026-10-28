
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Training & evaluation steps
# ---------------------------------------------------------------------------

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str,
               frozen_backbone: Optional[nn.Module] = None) -> Tuple[float, float]:

    model.train()
    if frozen_backbone is not None:
        frozen_backbone.eval()          # keep BN running stats frozen
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)                          # 1. Forward pass
        loss = loss_fn(y_pred, y)                  # 2. Tính loss

        optimizer.zero_grad()                      # 3. Reset gradient
        loss.backward()                            # 4. Backpropagation
        optimizer.step()                           # 5. Cập nhật tham số

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += torch.eq(y, y_pred.argmax(dim=1)).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training dataloader produced no samples.")
    return total_loss / total_samples, total_correct / total_samples


def evaluate_step(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  device: str) -> Tuple[float, float]:

    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            batch_size = y.size(0)
            total_loss += loss_fn(logits, y).item() * batch_size
            total_correct += torch.eq(y, logits.argmax(dim=1)).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Evaluation dataloader produced no samples.")
    return total_loss / total_samples, total_correct / total_samples


# ---------------------------------------------------------------------------
# Full training loop with optional TensorBoard logging
# ---------------------------------------------------------------------------

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: str,
          verbose: bool = True,
          writer=None,
          checkpoint_path: Optional[str] = None,
          frozen_backbone: Optional[nn.Module] = None) -> Dict[str, List[float]]:

    if epochs <= 0:
        raise ValueError(f"epochs must be positive; got {epochs}.")

    model.to(device)
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = float("-inf")

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, optimizer, device,
            frozen_backbone=frozen_backbone,
        )
        val_loss, val_acc = evaluate_step(model, val_dataloader, loss_fn, device)

        if verbose:
            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
            )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if checkpoint_path is not None:
                checkpoint_file = Path(checkpoint_path)
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_file)

        if writer is not None:
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train": train_loss, "validation": val_loss},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={"train": train_acc, "validation": val_acc},
                global_step=epoch,
            )

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    return results


# ---------------------------------------------------------------------------
# Quick evaluation helper
# ---------------------------------------------------------------------------

def eval_model(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: str = "cpu") -> Dict[str, object]:

    model.to(device)
    loss, acc = evaluate_step(model, dataloader, loss_fn, device)
    return {
        "model_name": model.__class__.__name__,
        "model_loss": round(loss, 4),
        "model_acc":  round(acc,  4),
    }
