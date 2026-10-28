"""
Contains utility functions: seeding, device selection, metrics,
timing, and saving/loading PyTorch models.
"""
import random
from pathlib import Path
from timeit import default_timer as timer

import torch


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
