
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

Device = Union[str, torch.device]


# ----------------------------------------------------------------------------
# 1. Figure helper
# ----------------------------------------------------------------------------
def finalize_figure(fig: Figure, save_path: Optional[str] = None,
                    show: bool = False, close: bool = True) -> Figure:
   
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved figure to: {save_path}")
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig


# ----------------------------------------------------------------------------
# 2. Inference (one pass over the data)
# ----------------------------------------------------------------------------
@dataclass
class InferenceResult:
    """Output of one inference pass. All tensors live on the CPU."""
    logits: torch.Tensor            # [N, num_classes]
    y_true: torch.Tensor            # [N]
    loss: Optional[float] = None    # mean loss over all N samples (if loss_fn given)

    @property
    def y_pred(self) -> torch.Tensor:
        return self.logits.argmax(dim=1)

    def probs(self) -> torch.Tensor:
        """Softmax probabilities, rows sum to 1."""
        return torch.softmax(self.logits, dim=1)


def run_inference(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  device: Device = "cpu",
                  loss_fn: Optional[Callable] = None) -> InferenceResult:

    model.to(device)
    model.eval()
    logits_all, targets = [], []
    is_cuda = torch.device(device).type == "cuda" if not isinstance(device, torch.device) else device.type == "cuda"
    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device, non_blocking=is_cuda)
            logits_all.append(model(X).cpu())
            targets.append(y.cpu())
    if not logits_all:
        raise ValueError("The dataloader is empty.")
    logits = torch.cat(logits_all)
    y_true = torch.cat(targets)
    loss = loss_fn(logits, y_true).item() if loss_fn is not None else None
    return InferenceResult(logits=logits, y_true=y_true, loss=loss)


def predict_probs(model: torch.nn.Module, images: torch.Tensor, device: Device = "cpu") -> torch.Tensor:

    model.to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1)
    return probs.cpu()


def predict_dataloader(model: torch.nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       device: Device = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Predicts the class of every sample. Kept for backward compatibility.

    Returns:
        A tuple (y_pred, y_true) of 1-D CPU tensors with class indices.
    """
    result = run_inference(model, dataloader, device)
    return result.y_pred, result.y_true


# ----------------------------------------------------------------------------
# 3. Confusion matrix and classification metrics
# ----------------------------------------------------------------------------
def compute_confusion_matrix(*, y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:

    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must be 1-D tensors of equal length")
    if y_true.numel() > 0:
        low = min(y_true.min().item(), y_pred.min().item())
        high = max(y_true.max().item(), y_pred.max().item())
        if low < 0 or high >= num_classes:
            raise ValueError(f"Labels must be in [0, {num_classes - 1}], got range [{low}, {high}]")
    idx = y_true * num_classes + y_pred
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def classification_metrics(cm: torch.Tensor) -> Dict[str, Union[float, torch.Tensor]]:

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm must be a square 2-D tensor")
    cm = cm.float()
    tp = cm.diag()
    support = cm.sum(dim=1)
    predicted = cm.sum(dim=0)
    precision = tp / predicted.clamp(min=1)
    recall = tp / support.clamp(min=1)
    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom.clamp(min=1e-12),
                     torch.zeros_like(denom))
    return {
        "accuracy": (tp.sum() / cm.sum().clamp(min=1)).item(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.long(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
    }


def per_class_accuracy(cm: torch.Tensor) -> torch.Tensor:
    """Returns the accuracy of each class (identical to per-class recall)."""
    return cm.diag().float() / cm.sum(dim=1).clamp(min=1).float()


def top_confusions(cm: torch.Tensor, class_names: Sequence[str], k: int = 5) -> List[Tuple[str, str, int]]:
    """Lists the most frequent mistakes as (true_class, predicted_class, count).

    Cells with a count of 0 are never listed, so fewer than k items may return.
    """
    if len(class_names) != cm.shape[0]:
        raise ValueError("class_names length must match the confusion matrix size")
    off_diag = cm.clone()
    off_diag.fill_diagonal_(0)
    k = min(k, off_diag.numel())
    if k <= 0:
        return []
    values, indices = off_diag.flatten().topk(k)
    n = len(class_names)
    results = []
    for value, index in zip(values.tolist(), indices.tolist()):
        if value == 0:  # topk is sorted descending, so the rest are zeros too
            break
        true_idx, pred_idx = divmod(index, n)
        results.append((class_names[true_idx], class_names[pred_idx], int(value)))
    return results


def wilson_interval(correct: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for an accuracy (default 95%).

    Raises:
        ValueError: If total <= 0 or correct is outside [0, total].
    """
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("Need total > 0 and 0 <= correct <= total")
    p = correct / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


# ----------------------------------------------------------------------------
# 4. Error analysis
# ----------------------------------------------------------------------------
def misclassified_indices(y_true: torch.Tensor, y_pred: torch.Tensor,
                          logits: Optional[torch.Tensor] = None,
                          max_items: Optional[int] = None) -> torch.Tensor:

    wrong = (y_pred != y_true).nonzero().flatten()
    if logits is not None and wrong.numel() > 0:
        conf = torch.softmax(logits[wrong], dim=1).max(dim=1).values
        wrong = wrong[conf.argsort(descending=True)]
    if max_items is not None:
        wrong = wrong[:max_items]
    return wrong


def get_mean_std(transform, transform_weights=None) -> Tuple[List[float], List[float]]:

    for t in list(getattr(transform, "transforms", [transform])):
        mean, std = getattr(t, "mean", None), getattr(t, "std", None)
        if mean is not None and std is not None:
            return [float(m) for m in mean], [float(s) for s in std]

    # Fallback: read stats from the preprocessing preset of the weights enum.
    if transform_weights is not None:
        preprocessing = getattr(transform_weights, "preprocessing", None)
        if preprocessing is not None:
            mean, std = getattr(preprocessing, "mean", None), getattr(preprocessing, "std", None)
            if mean is not None and std is not None:
                return [float(m) for m in mean], [float(s) for s in std]

    raise ValueError("No mean/std found in the given transform and no valid transform_weights fallback")


def denormalize(images: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """Inverts channel normalization and clamps to [0, 1] for display.

    Args:
        images: Tensor [C, H, W] or [N, C, H, W].
    """
    if len(mean) != len(std) or images.ndim not in (3, 4) or images.shape[-3] != len(mean):
        raise ValueError("images must be [C,H,W] or [N,C,H,W] with C == len(mean) == len(std)")
    mean_t = torch.tensor(mean, dtype=images.dtype, device=images.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=images.dtype, device=images.device).view(-1, 1, 1)
    return (images * std_t + mean_t).clamp(0.0, 1.0)


# ----------------------------------------------------------------------------
# 5. Cost measurement
# ----------------------------------------------------------------------------
def measure_inference_time(model: torch.nn.Module, images: torch.Tensor,
                           device: Device = "cpu", warmup: int = 3, repeats: int = 10) -> Dict:

    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")
    device = torch.device(device)
    is_cuda = device.type == "cuda"
    model.to(device)
    model.eval()
    x = images.to(device)

    def sync() -> None:
        if is_cuda:
            torch.cuda.synchronize(device)

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            model(x)
            sync()
            times.append(time.perf_counter() - start)
    median = statistics.median(times)
    return {
        "batch_size": int(x.shape[0]),
        "ms_per_batch": median * 1000,
        "ms_per_image": median * 1000 / x.shape[0],
        "repeats": repeats,
        "device": torch.cuda.get_device_name(device) if is_cuda else str(device),
        "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 2 ** 20 if is_cuda else None,
    }


def file_size_mb(path: str) -> float:
    """Size of a file in MiB (raises FileNotFoundError if missing)."""
    return Path(path).stat().st_size / 2 ** 20


def build_cost_record(num_params: Dict[str, int], model_path: Optional[str] = None,
                      timing: Optional[Dict] = None, history: Optional[Dict] = None,
                      source_history: Optional[Dict] = None) -> Dict:

    timing = timing or {}
    train_time_phase = None
    if history and history.get("epoch_time"):
        train_time_phase = float(sum(history["epoch_time"]))

    train_time_total = train_time_phase
    if source_history and source_history.get("epoch_time"):
        source_total = float(sum(source_history["epoch_time"]))
        if train_time_total is None:
            train_time_total = source_total
        else:
            train_time_total = train_time_total + source_total

    return {
        "num_params_total": num_params.get("total"),
        "num_params_trainable": num_params.get("trainable"),
        "model_file_mb": file_size_mb(model_path) if model_path is not None else None,
        "ms_per_image": timing.get("ms_per_image"),
        "ms_per_batch": timing.get("ms_per_batch"),
        "timing_batch_size": timing.get("batch_size"),
        "peak_gpu_mb": timing.get("peak_gpu_mb"),
        "device": timing.get("device"),
        "train_time_phase_s": train_time_phase,
        "train_time_total_s": train_time_total,
    }


# ----------------------------------------------------------------------------
# 6. Summaries and comparison
# ----------------------------------------------------------------------------
def evaluate_predictions(result: InferenceResult, class_names: Sequence[str]) -> Dict:

    n = len(class_names)
    cm = compute_confusion_matrix(y_true=result.y_true, y_pred=result.y_pred, num_classes=n)
    m = classification_metrics(cm)
    total, correct = int(cm.sum()), int(cm.diag().sum())
    low, high = wilson_interval(correct, total)
    probs = result.probs()
    confidence = probs.max(dim=1).values.tolist()
    return {
        "num_samples": total,
        "loss": result.loss,
        "accuracy": m["accuracy"],
        "accuracy_ci95": [low, high],
        "macro_precision": m["macro_precision"],
        "macro_recall": m["macro_recall"],
        "macro_f1": m["macro_f1"],
        "per_class": {
            name: {
                "precision": m["precision"][i].item(),
                "recall": m["recall"][i].item(),
                "f1": m["f1"][i].item(),
                "support": int(m["support"][i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
        "predictions": {
            "y_true": result.y_true.tolist(),
            "y_pred": result.y_pred.tolist(),
            "confidence": confidence,
        },
    }


def _fmt(value, template: str) -> str:
    return "—" if value is None else template.format(value)


def format_comparison_table(runs: Dict[str, Dict]) -> str:

    header = ("| Run | Val acc | Test acc | 95% CI | Macro F1 | Params | Trainable | "
              "Model (MB) | ms/image | Train time phase (s) | Train time total (s) |")
    lines = [header, "|" + " :--- |" * 11]
    for name, run in runs.items():
        m, c = run["metrics"], run.get("cost") or {}
        low, high = m["accuracy_ci95"]
        val_acc = run.get("best_val_acc")
        val_text = _fmt(None if val_acc is None else val_acc * 100, "{:.2f}%")
        lines.append(
            f"| {name} "
            f"| {val_text} "
            f"| {m['accuracy'] * 100:.2f}% "
            f"| [{low * 100:.2f}, {high * 100:.2f}] "
            f"| {m['macro_f1']:.4f} "
            f"| {_fmt(c.get('num_params_total'), '{:,}')} "
            f"| {_fmt(c.get('num_params_trainable'), '{:,}')} "
            f"| {_fmt(c.get('model_file_mb'), '{:.1f}')} "
            f"| {_fmt(c.get('ms_per_image'), '{:.2f}')} "
            f"| {_fmt(c.get('train_time_phase_s'), '{:.0f}')} "
            f"| {_fmt(c.get('train_time_total_s'), '{:.0f}')} |"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 7. Plots (all return the Figure)
# ----------------------------------------------------------------------------
def plot_confusion_matrix(cm: torch.Tensor, class_names: Sequence[str],
                          title: str = "Confusion matrix", normalize: bool = False,
                          save_path: Optional[str] = None, show: bool = False) -> Figure:
    """Plots a confusion matrix; normalize=True shows each row as fractions."""
    n = len(class_names)
    if cm.shape != (n, n):
        raise ValueError("class_names length must match the confusion matrix size")
    data = cm.float() / cm.sum(dim=1, keepdim=True).clamp(min=1) if normalize else cm
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(data.numpy(), cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    threshold = data.max().item() / 2
    for i in range(n):
        for j in range(n):
            value = data[i, j].item()
            text = f"{value:.2f}" if normalize else str(int(value))
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if value > threshold else "black")
    fig.tight_layout()
    return finalize_figure(fig, save_path, show)


def plot_per_class_accuracy(values: torch.Tensor, class_names: Sequence[str],
                            title: str = "Per-class accuracy",
                            save_path: Optional[str] = None, show: bool = False) -> Figure:
    """Bar chart of a per-class value in [0, 1] (e.g. recall)."""
    if len(values) != len(class_names):
        raise ValueError("values and class_names must have the same length")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(range(len(class_names)), values.numpy())
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values.tolist()):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return finalize_figure(fig, save_path, show)


def plot_sample_predictions(images: Sequence[torch.Tensor], true_labels: Sequence[int],
                            pred_labels: Sequence[int], class_names: Sequence[str],
                            ncols: int = 3, save_path: Optional[str] = None,
                            show: bool = False) -> Figure:

    n = len(images)
    if n == 0:
        raise ValueError("No images to plot")
    if not (len(true_labels) == len(pred_labels) == n):
        raise ValueError("images, true_labels and pred_labels must have the same length")
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows), squeeze=False)
    for i, ax in enumerate(axes.flatten()):
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(images[i].permute(1, 2, 0).numpy())  # [C, H, W] -> [H, W, C]
        pred, truth = class_names[pred_labels[i]], class_names[true_labels[i]]
        ok = pred == truth
        ax.set_title(f"{'✓' if ok else '✗'} Pred: {pred}\nTruth: {truth}", fontsize=10,
                     color="g" if ok else "r")
        ax.axis("off")
    fig.tight_layout()
    return finalize_figure(fig, save_path, show)


def plot_history(histories: Dict[str, Dict[str, List[float]]],
                 start_epochs: Optional[Dict[str, int]] = None,
                 save_path: Optional[str] = None, show: bool = False) -> Figure:


    start_epochs = start_epochs or {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, h in histories.items():
        start = start_epochs.get(name, 0)
        epochs = range(start + 1, start + len(h["train_loss"]) + 1)

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
    return finalize_figure(fig, save_path, show)
