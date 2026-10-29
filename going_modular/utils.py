"""
Contains utility functions: seeding, device selection, metrics,
timing, and saving/loading PyTorch models.
"""
import json
import random
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, List, Tuple

import torch


HISTORY_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc")


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility.

    Args:
        seed: Integer seed used for Python's random module and PyTorch.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Returns "cuda" if a GPU is available, otherwise "cpu"."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculates accuracy between true and predicted labels.

    Args:
        y_true: Tensor of true class indices, shape [N].
        y_pred: Tensor of predicted class indices, shape [N].

    Returns:
        Accuracy as a fraction between 0 and 1 (multiply by 100 for percent).
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    return correct / len(y_pred)


def print_train_time(start: float, end: float, device=None) -> float:
    """Prints and returns the elapsed training time in seconds.

    Args:
        start: Start time from timeit.default_timer().
        end: End time from timeit.default_timer().
        device: Optional device name, only used for printing.
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def save_model(model: torch.nn.Module, target_dir: str, model_name: str) -> Path:
    """Saves a PyTorch model's state_dict to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename ending in ".pth" or ".pt".

    Returns:
        The path the model was saved to.

    Example usage:
        save_model(model=model, target_dir="models", model_name="cifar10_v2.pth")
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
        "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
    return model_save_path


def load_model(model: torch.nn.Module, model_path: str, device: str = "cpu") -> torch.nn.Module:
    """Loads a saved state_dict into an existing model instance.

    The model must be created with the same architecture as the saved one.

    Args:
        model: An instance of the model architecture to load weights into.
        model_path: Path to the saved .pth/.pt file.
        device: Device to map the weights onto.

    Returns:
        The model with loaded weights, moved to `device`.
    """
    state_dict = torch.load(f=model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)


def save_training_metadata(
    history: Dict[str, List[float]],
    artifact_dir: str,
    config: Dict[str, object],
) -> Tuple[Path, Path]:
    """Persist training curves and their configuration beside a model checkpoint."""
    missing_keys = set(HISTORY_KEYS) - history.keys()
    if missing_keys:
        raise ValueError(f"Training history is missing keys: {sorted(missing_keys)}")
    if not history["train_loss"]:
        raise ValueError("Training history is empty; refusing to save incomplete results.")
    if any(len(history[key]) != len(history["train_loss"]) for key in HISTORY_KEYS):
        raise ValueError("All training history series must have the same number of epochs.")

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    history_path = artifact_path / "history.json"
    config_path = artifact_path / "config.json"
    history_tmp = artifact_path / "history.json.tmp"
    config_tmp = artifact_path / "config.json.tmp"

    history_tmp.write_text(json.dumps(history, indent=2), encoding="utf-8")
    config_tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
    history_tmp.replace(history_path)
    config_tmp.replace(config_path)
    return history_path, config_path


def load_training_artifacts(
    model: torch.nn.Module,
    artifact_dir: str,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, List[float]], Dict[str, object]]:
    """Load a model checkpoint, its training curves, and saved run configuration."""
    artifact_path = Path(artifact_dir)
    checkpoint_path = artifact_path / "best_model.pth"
    history_path = artifact_path / "history.json"
    config_path = artifact_path / "config.json"
    required_paths = (checkpoint_path, history_path, config_path)
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Training artifacts are incomplete. Missing: "
            + ", ".join(missing_paths)
            + ". Run the notebook with RUN_MODE='train' first."
        )

    loaded_model = load_model(model, str(checkpoint_path), device=device)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing_keys = set(HISTORY_KEYS) - history.keys()
    if missing_keys:
        raise ValueError(f"Saved history is missing keys: {sorted(missing_keys)}")
    if not history["train_loss"]:
        raise ValueError(f"Saved training history is empty: {history_path}")
    if any(len(history[key]) != len(history["train_loss"]) for key in HISTORY_KEYS):
        raise ValueError(f"Saved training history has inconsistent lengths: {history_path}")
    return loaded_model, history, config
