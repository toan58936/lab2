
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# History key schemas
# ---------------------------------------------------------------------------

HISTORY_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc")
HISTORY_KEYS_SIX = ("train_loss", "val_loss", "train_acc", "val_acc", "lr", "epoch_time")

ARTIFACT_REQUIRED = ("best_model.pth", "history.json", "config.json")


# ---------------------------------------------------------------------------
# Seed and device
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Sets random seeds for reproducibility.

    Seeds Python's random, NumPy, and PyTorch (CPU and CUDA when available).
    When ``deterministic`` is True, sets PyTorch/CUDA backend flags for
    deterministic behavior where supported.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(prefer: Optional[str] = None) -> torch.device:

    if prefer is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    prefer = prefer.lower()
    valid = ("cuda", "mps", "cpu")
    if prefer not in valid:
        raise ValueError(
            f"Invalid device preference '{prefer}'. Choose one of {valid}."
        )

    if prefer == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available on this system.")
    if prefer == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS requested but not available on this system.")

    return torch.device(prefer)


def get_env_info(device: torch.device) -> Dict[str, str]:
    """Return device and library version info for config/logging."""
    info: Dict[str, str] = {
        "device": str(device),
        "torch": torch.__version__,
    }
    try:
        import torchvision  # type: ignore

        info["torchvision"] = torchvision.__version__
    except ImportError:
        info["torchvision"] = "not installed"
    try:
        import numpy as np  # type: ignore

        info["numpy"] = np.__version__
    except ImportError:
        info["numpy"] = "not installed"
    return info


def ensure_compatible_state_dict(
    state_dict: dict, arch: str | None = None, source_path: str | None = None
) -> None:
    """Từ chối checkpoint/state_dict format cũ (wrapper backbone/head) theo hướng B.

    Không migrate key. Yêu cầu train lại 3 run với format native torchvision.
    """
    keys = list(state_dict.keys()) if isinstance(state_dict, dict) else []
    has_legacy = any(
        isinstance(k, str) and (k.startswith("backbone.") or k.startswith("head."))
        for k in keys
    )
    if not has_legacy:
        return

    arch_hint = ""
    if arch == "resnet18":
        arch_hint = " (mong đợi key native như conv1.*, layer*, fc.*)"
    elif arch == "vgg16":
        arch_hint = " (mong đợi key native như features.*, classifier.*)"

    src = f" từ '{source_path}'" if source_path else ""
    raise ValueError(
        "Checkpoint/state_dict không tương thích với format model native hiện tại"
        f"{src}{arch_hint}: phát hiện key kiểu cũ 'backbone.*'/'head.*' (wrapper). "
        "Không hỗ trợ migrate. Hãy đặt FORCE_RETRAIN=True hoặc dọn artifacts của run "
        "(v0_vgg16, v1_resnet18, v1_resnet18_finetune) rồi train lại theo format mới."
    )


def load_model(
    model: torch.nn.Module, model_path: str, device: str = "cpu", arch: str | None = None
) -> torch.nn.Module:
    """Loads a saved state_dict into an existing model instance.

    Uses ``weights_only=True`` for safe loading. Does not fall back to unsafe
    loading if the payload is incompatible.
    """
    state_dict = torch.load(f=model_path, map_location=device, weights_only=True)
    ensure_compatible_state_dict(state_dict, arch=arch, source_path=model_path)
    model.load_state_dict(state_dict)
    return model.to(device)


# ---------------------------------------------------------------------------
# Phase 4: atomic JSON helpers
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON to a temp file then atomically replace the target.

    If writing fails, the original target file is left intact and the temp
    file is removed. The error is re-raised.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _read_json(path: Path) -> Any:
    """Read and parse a JSON file with UTF-8 encoding."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_model(model: torch.nn.Module, path: str) -> Path:
    """Lưu state_dict của mô hình vào file ``path`` (atomic).

    Args:
        model: PyTorch model.
        path: Đường dẫn file đích (ví dụ ``artifacts/run/best_model.pth``).

    Returns:
        Path của file đã lưu.
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    try:
        torch.save(obj=model.state_dict(), f=tmp_path)
        os.replace(tmp_path, save_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return save_path


# ---------------------------------------------------------------------------
# Phase 4: config / history / metrics
# ---------------------------------------------------------------------------

def save_config(artifact_dir: str, config: Dict[str, object]) -> Path:
    """Atomically save run configuration to ``config.json``.

    Validates that ``config`` is a non-empty dict before writing.
    """
    if not isinstance(config, dict) or not config:
        raise ValueError("config must be a non-empty dict.")
    artifact_path = Path(artifact_dir)
    config_path = artifact_path / "config.json"
    _atomic_write_json(config_path, config)
    return config_path


def save_history(artifact_dir: str, history: Dict[str, List[float]]) -> Path:
    """Atomically save training history to ``history.json``.

    Validates the six required history keys: train_loss, val_loss,
    train_acc, val_acc, lr, epoch_time. All series must have equal length
    and at least one epoch.
    """
    missing_keys = set(HISTORY_KEYS_SIX) - history.keys()
    if missing_keys:
        raise ValueError(
            f"History is missing keys: {sorted(missing_keys)}. "
            f"Required: {list(HISTORY_KEYS_SIX)}."
        )
    if not history["train_loss"]:
        raise ValueError("History is empty; refusing to save incomplete results.")
    n = len(history["train_loss"])
    for key in HISTORY_KEYS_SIX:
        if len(history[key]) != n:
            raise ValueError(
                f"All history series must have the same length; "
                f"'{key}' has {len(history[key])}, expected {n}."
            )

    artifact_path = Path(artifact_dir)
    history_path = artifact_path / "history.json"
    _atomic_write_json(history_path, history)
    return history_path


def load_experiment(artifact_dir: str) -> Tuple[Dict[str, List[float]], Dict[str, object]]:
    """Load ``history.json`` and ``config.json`` from an artifact directory.

    Raises ``FileNotFoundError`` if either file is missing and
    ``ValueError`` if the payload is invalid.
    """
    artifact_path = Path(artifact_dir)
    history_path = artifact_path / "history.json"
    config_path = artifact_path / "config.json"

    if not history_path.is_file():
        raise FileNotFoundError(f"Missing artifact: {history_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing artifact: {config_path}")

    history = _read_json(history_path)
    config = _read_json(config_path)

    missing_keys = set(HISTORY_KEYS_SIX) - history.keys()
    if missing_keys:
        raise ValueError(f"Saved history is missing keys: {sorted(missing_keys)}")

    return history, config


def save_test_metrics(artifact_dir: str, metrics: Dict[str, object]) -> Path:
    """Atomically save test metrics to ``test_metrics.json``.

    Refuses to write empty metrics (no placeholder/eval-fake data).
    """
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("metrics must be a non-empty dict; will not write empty metrics.")
    artifact_path = Path(artifact_dir)
    metrics_path = artifact_path / "test_metrics.json"
    _atomic_write_json(metrics_path, metrics)
    return metrics_path


def load_test_metrics(artifact_dir: str) -> Dict[str, object]:
    """Load test metrics from ``test_metrics.json``.

    Raises ``FileNotFoundError`` if the file is missing.
    """
    artifact_path = Path(artifact_dir)
    metrics_path = artifact_path / "test_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing artifact: {metrics_path}")
    return _read_json(metrics_path)


# ---------------------------------------------------------------------------
# Phase 4: artifacts existence
# ---------------------------------------------------------------------------

def artifacts_exist(artifact_dir: str) -> bool:
    """Return True if all required training artifacts exist.

    Required: ``best_model.pth``, ``history.json``, ``config.json``.
    ``test_metrics.json`` is optional.
    """
    artifact_path = Path(artifact_dir)
    return all((artifact_path / name).is_file() for name in ARTIFACT_REQUIRED)


# ---------------------------------------------------------------------------
# Phase 4: TensorBoard writer
# ---------------------------------------------------------------------------

def create_writer(experiment_name: str, log_root: str = "runs") -> SummaryWriter:
    """Create a TensorBoard SummaryWriter at ``runs/<experiment_name>``.

    The caller is responsible for closing the writer (e.g. via try/finally).
    """
    log_dir = Path(log_root) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


# ---------------------------------------------------------------------------
# Phase 5: dated outputs and session logging
# ---------------------------------------------------------------------------

import hashlib  # noqa: E402  (module-level import placed near use)
from datetime import datetime, timezone  # noqa: E402

HOCHI_MINH_TZ = None

try:
    from zoneinfo import ZoneInfo
    HOCHI_MINH_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
except ImportError:  # pragma: no cover - Python < 3.9 fallback
    HOCHI_MINH_TZ = timezone(__import__("datetime").timedelta(hours=7))


def make_session_id(tz=HOCHI_MINH_TZ) -> str:
    """Tạo session ID duy nhất theo múi giờ đã cho (mặc định Asia/Ho_Chi_Minh).

    Định dạng: YYYYMMDD-HHMMSS+07:00. Đảm bảo các phiên trong cùng ngày
    không ghi đè nhau.
    """
    now = datetime.now(tz)
    return now.strftime("%Y%m%d-%H%M%S%z")


def make_eval_id(tz=HOCHI_MINH_TZ) -> str:
    """Tạo evaluation ID duy nhất (cùng định dạng session ID)."""
    return make_session_id(tz)


def dated_dir(root: str | Path, date_str: str | None = None, tz=HOCHI_MINH_TZ) -> Path:
    """Tạo thư mục theo ngày dưới root: root/YYYY-MM-DD/."""
    if date_str is None:
        now = datetime.now(tz)
        date_str = now.strftime("%Y-%m-%d")
    return Path(root) / date_str


def make_eval_output_dir(
    output_root: str | Path,
    run_id: str,
    date_str: str | None = None,
    eval_id: str | None = None,
    tz=HOCHI_MINH_TZ,
) -> Path:
    """Tạo thư mục đầu ra đánh giá: output_root/YYYY-MM-DD/<RUN_ID>/<EVAL_ID>/."""
    day_dir = dated_dir(output_root, date_str, tz)
    if eval_id is None:
        eval_id = make_eval_id(tz)
    return day_dir / run_id / eval_id


def make_session_log_dir(
    log_root: str | Path,
    run_id: str,
    session_id: str | None = None,
    tz=HOCHI_MINH_TZ,
) -> Path:
    """Tạo thư mục log TensorBoard cho phiên: log_root/<RUN_ID>/<SESSION_ID>/."""
    if session_id is None:
        session_id = make_session_id(tz)
    return Path(log_root) / run_id / session_id


def file_sha256(path: str | Path) -> str:
    """Tính mã băm SHA-256 của file, dùng cho cache đánh giá theo checkpoint."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Phase 4: checkpoint resume
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    phase_epochs_completed: int,
    best_score: float,
    best_epoch: int,
    history: Dict[str, List[float]],
    phase_config: Dict[str, object],
) -> Path:
    """Save a resume checkpoint containing model, optimizer, epoch, score and history.

    Only state dicts and primitive/JSON-compatible data are stored — no
    raw model, optimizer, or callable objects. Saved with ``torch.save`` so it
    can be loaded back with ``weights_only=True``.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "phase_epochs_completed": phase_epochs_completed,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "history": history,
        "phase_config": phase_config,
    }
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, checkpoint_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return checkpoint_path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    expected_phase_config: Dict[str, object],
    arch: str | None = None,
) -> Tuple[int, float, int, Dict[str, List[float]]]:
    """Load a resume checkpoint and verify phase config compatibility.

    Returns ``(phase_epochs_completed, best_score, best_epoch, history)``.

    Raises ``FileNotFoundError`` if the checkpoint is missing and
    ``ValueError`` if the phase config does not match.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    payload = torch.load(str(checkpoint_path), map_location=str(device), weights_only=True)

    saved_phase_config = payload.get("phase_config", {})
    if saved_phase_config != expected_phase_config:
        raise ValueError(
            "Phase config mismatch: checkpoint was created with a different "
            "run configuration. Resume refused."
        )

    model_state = payload["model_state_dict"]
    ensure_compatible_state_dict(model_state, arch=arch, source_path=str(checkpoint_path))
    model.load_state_dict(model_state)
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    return (
        payload["phase_epochs_completed"],
        payload["best_score"],
        payload["best_epoch"],
        payload["history"],
    )
