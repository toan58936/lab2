"""
Contains functions for evaluating a trained PyTorch model: predictions,
confusion matrix, per-class accuracy and plotting helpers.
Figures are saved to disk (and optionally shown) so scripts also work
without an interactive display.
"""
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch


def save_or_show(save_path: Optional[str] = None, show: bool = False) -> None:
    """Saves the current matplotlib figure and/or shows it, then closes it.

    Args:
        save_path: Where to save the figure. None means do not save.
        show: Whether to open an interactive window.
    """
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved figure to: {save_path}")
    if show:
        plt.show()
    plt.close()


def predict_probs(model: torch.nn.Module, images: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Returns softmax probabilities for a batch of images.

    Args:
        model: A trained PyTorch model.
        images: Tensor of shape [N, C, H, W].
        device: Device to run the model on.

    Returns:
        A CPU tensor of shape [N, num_classes] whose rows sum to 1.
    """
    model.to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1)
    return probs.cpu()


def predict_dataloader(model: torch.nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Predicts the class of every sample in a DataLoader.

    Args:
        model: A trained PyTorch model.
        dataloader: A DataLoader (use shuffle=False so order is stable).
        device: Device to run the model on.

    Returns:
        A tuple (y_pred, y_true) of 1-D CPU tensors with class indices.
    """
    model.to(device)
    model.eval()
    preds, targets = [], []
    with torch.inference_mode():
        for X, y in dataloader:
            logits = model(X.to(device))
            preds.append(logits.argmax(dim=1).cpu())
            targets.append(y.cpu())
    return torch.cat(preds), torch.cat(targets)


def compute_confusion_matrix(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Builds a confusion matrix: rows are true classes, columns are predictions.

    Args:
        y_pred: 1-D tensor of predicted class indices.
        y_true: 1-D tensor of true class indices.
        num_classes: Number of classes.

    Returns:
        An integer tensor of shape [num_classes, num_classes].
    """
    idx = y_true * num_classes + y_pred
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def per_class_accuracy(cm: torch.Tensor) -> torch.Tensor:
    """Returns the accuracy of each class (fraction between 0 and 1)."""
    return cm.diag().float() / cm.sum(dim=1).clamp(min=1).float()


def top_confusions(cm: torch.Tensor, class_names: Sequence[str], k: int = 5) -> List[Tuple[str, str, int]]:
    """Lists the most frequent mistakes as (true_class, predicted_class, count)."""
    off_diag = cm.clone()
    off_diag.fill_diagonal_(0)
    k = min(k, off_diag.numel())
    values, indices = off_diag.flatten().topk(k)
    n = len(class_names)
    results = []
    for value, index in zip(values.tolist(), indices.tolist()):
        true_idx, pred_idx = divmod(index, n)
        results.append((class_names[true_idx], class_names[pred_idx], int(value)))
    return results


def plot_confusion_matrix(cm: torch.Tensor, class_names: Sequence[str],
                          title: str = "Confusion matrix",
                          save_path: Optional[str] = None, show: bool = False) -> None:
    """Plots a confusion matrix with the count written in every cell."""
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm.numpy(), cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    threshold = cm.max().item() / 2
    for i in range(n):
        for j in range(n):
            value = cm[i, j].item()
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8,
                    color="white" if value > threshold else "black")
    fig.tight_layout()
    save_or_show(save_path, show)


def plot_sample_predictions(images: Sequence[torch.Tensor], true_labels: Sequence[int],
                            pred_labels: Sequence[int], class_names: Sequence[str],
                            ncols: int = 3, save_path: Optional[str] = None,
                            show: bool = False) -> None:
    """Plots a grid of color images. Title is green if correct, red if wrong.

    Args:
        images: List of tensors of shape [3, H, W] with values in [0, 1].
        true_labels: True class indices.
        pred_labels: Predicted class indices.
        class_names: Names of the classes.
        ncols: Number of columns in the grid.
        save_path: Where to save the figure (optional).
        show: Whether to open an interactive window.
    """
    n = len(images)
    nrows = math.ceil(n / ncols)
    plt.figure(figsize=(3.2 * ncols, 3.4 * nrows))
    for i in range(n):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))  # [C, H, W] -> [H, W, C]
        pred, truth = class_names[pred_labels[i]], class_names[true_labels[i]]
        plt.title(f"Pred: {pred}\nTruth: {truth}", fontsize=10,
                  color="g" if pred == truth else "r")
        plt.axis(False)
    plt.tight_layout()
    save_or_show(save_path, show)


def plot_history(histories: Dict[str, Dict[str, List[float]]],
                 save_path: Optional[str] = None, show: bool = False) -> None:
    """Plots loss and accuracy curves (validation = solid, train = dashed).

    Args:
        histories: Mapping of run name to the dict returned by engine.train().
        save_path: Where to save the figure (optional).
        show: Whether to open an interactive window.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, h in histories.items():
        epochs = range(1, len(h["train_loss"]) + 1)

        (line,) = axes[0].plot(epochs, h["val_loss"], label=f"{name} (validation)")
        axes[0].plot(epochs, h["train_loss"], "--", color=line.get_color(), label=f"{name} (train)")

        (line,) = axes[1].plot(epochs, h["val_acc"], label=f"{name} (validation)")
        axes[1].plot(epochs, h["train_acc"], "--", color=line.get_color(), label=f"{name} (train)")

    axes[0].set_title("Loss")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_or_show(save_path, show)
