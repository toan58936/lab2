"""Kiểm thử cho going_modular.utils (Phase 4).

Chạy nhanh (không cần GPU/mạng):
    pytest tests/test_utils.py -v

Ánh xạ mã test theo test spec: U1–U11.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn, optim

from going_modular import utils
from going_modular.utils import (
    artifacts_exist,
    create_writer,
    get_device,
    get_env_info,
    load_checkpoint,
    load_experiment,
    load_model,
    load_test_metrics,
    save_checkpoint,
    save_config,
    save_history,
    save_model,
    save_test_metrics,
    set_seed,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# U1: Seed reproducibility
# ---------------------------------------------------------------------------

def test_u1_seed_reproducible():
    """U1: Cùng seed hai lần → chuỗi số Python/NumPy/PyTorch lặp lại."""
    import random
    import numpy as np

    set_seed(42)
    py_vals = [random.random() for _ in range(5)]
    np_vals = np.random.random(5).tolist()
    torch_vals = torch.randn(5).tolist()

    set_seed(42)
    assert [random.random() for _ in range(5)] == py_vals
    assert np.random.random(5).tolist() == np_vals
    assert torch.randn(5).tolist() == torch_vals


# ---------------------------------------------------------------------------
# U2: Device selection
# ---------------------------------------------------------------------------

def test_u2_cpu_device():
    """U2: Không chỉ định prefer → trả về torch.device (cpu trên máy không GPU)."""
    dev = get_device()
    assert isinstance(dev, torch.device)
    assert dev.type in ("cpu", "cuda", "mps")


def test_u2_prefer_cpu():
    dev = get_device(prefer="cpu")
    assert dev == torch.device("cpu")


def test_u2_invalid_prefer():
    with pytest.raises(ValueError):
        get_device(prefer="xpu")


def test_u2_unavailable_device():
    if torch.cuda.is_available():
        pytest.skip("CUDA có sẵn, không thể kiểm tra thiết bị không khả dụng.")
    with pytest.raises(ValueError):
        get_device(prefer="cuda")


# ---------------------------------------------------------------------------
# U3: Config / history JSON round-trip with Unicode
# ---------------------------------------------------------------------------

def test_u3_config_history_roundtrip(tmp_path):
    """U3: Lưu/nạp JSON có Unicode, config/history theo schema → round-trip giữ dữ liệu."""
    config = {"model": "resnet18", "mô_hình": "ResNet-18 Tiếng Việt"}
    save_config(tmp_path, config)
    loaded_config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert loaded_config == config

    history = {
        "train_loss": [0.5, 0.4],
        "val_loss": [0.6, 0.5],
        "train_acc": [0.8, 0.85],
        "val_acc": [0.78, 0.82],
        "lr": [1e-3, 1e-3],
        "epoch_time": [10.0, 10.5],
    }
    save_history(tmp_path, history)
    loaded_history, _ = load_experiment(tmp_path)
    assert loaded_history == history


def test_u3_empty_config_rejected():
    """U3: Config rỗng → ValueError."""
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            save_config(d, {})


# ---------------------------------------------------------------------------
# U4: Model state dict round-trip
# ---------------------------------------------------------------------------

def test_u4_model_state_dict_roundtrip(tmp_path):
    """U4: Lưu/nạp state dict → cùng kiến trúc/input cho cùng đầu ra."""
    model = TinyModel()
    x = torch.randn(2, 4)
    with torch.no_grad():
        out_before = model(x)

    path = tmp_path / "model.pth"
    save_model(model, str(path))

    model2 = TinyModel()
    load_model(model2, str(path), device="cpu")
    with torch.no_grad():
        out_after = model2(x)

    assert torch.equal(out_before, out_after)


# ---------------------------------------------------------------------------
# U5: Atomic write failure keeps original and cleans temp
# ---------------------------------------------------------------------------

def test_u5_atomic_write_keeps_original_on_failure(tmp_path, monkeypatch):
    """U5: Giả lập os.replace lỗi sau khi temp đã ghi → file đích nguyên vẹn, temp dọn, lỗi truyền lên."""
    target = Path(tmp_path) / "config.json"
    target.write_text('{"original": true}', encoding="utf-8")

    original_replace = os.replace

    def failing_replace(src, dst):
        raise OSError("disk full on replace")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError, match="disk full on replace"):
        save_config(tmp_path, {"new": True})

    assert target.read_text(encoding="utf-8") == '{"original": true}'
    # Temp file phải được dọn sạch
    tmp_files = list(Path(tmp_path).glob("*.tmp"))
    assert len(tmp_files) == 0, f"Temp files còn lại: {tmp_files}"


# ---------------------------------------------------------------------------
# U6: Artifacts existence
# ---------------------------------------------------------------------------

def test_u6_artifacts_exist_all_present(tmp_path):
    """U6: Đủ best_model.pth, history.json, config.json → artifacts_exist True."""
    (tmp_path / "best_model.pth").write_bytes(b"x")
    (tmp_path / "history.json").write_text("{}", encoding="utf-8")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert artifacts_exist(tmp_path) is True


def test_u6_artifacts_missing_each(tmp_path):
    """U6: Thiếu lần lượt từng artifact bắt buộc → artifacts_exist False."""
    assert artifacts_exist(tmp_path) is False
    (tmp_path / "best_model.pth").write_bytes(b"x")
    assert artifacts_exist(tmp_path) is False
    (tmp_path / "history.json").write_text("{}", encoding="utf-8")
    assert artifacts_exist(tmp_path) is False
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert artifacts_exist(tmp_path) is True


# ---------------------------------------------------------------------------
# U7: Missing file / bad schema errors
# ---------------------------------------------------------------------------

def test_u7_missing_artifact_raises(tmp_path):
    """U7: Thiếu file → FileNotFoundError nêu tên file thiếu."""
    with pytest.raises(FileNotFoundError):
        load_experiment(tmp_path)


def test_u7_bad_schema_raises(tmp_path):
    """U7: history sai schema → ValueError."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "history.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment(tmp_path)


# ---------------------------------------------------------------------------
# U8: Safe loading rejects unsafe payload
# ---------------------------------------------------------------------------

def test_u8_unsafe_payload_rejected(tmp_path):
    """U8: Payload .pth chứa đối tượng Python tùy ý → bị từ chối, không fallback."""
    # Tạo file chứa một đối tượng Python (không phải state_dict)
    bad_path = tmp_path / "bad.pth"
    torch.save({"model_state_dict": {"fc.weight": torch.randn(2, 4)}, "extra": object()}, bad_path)

    model = TinyModel()
    with pytest.raises((TypeError, RuntimeError, KeyError, pickle.UnpicklingError)):
        load_model(model, str(bad_path), device="cpu")


# ---------------------------------------------------------------------------
# U9: TensorBoard writer
# ---------------------------------------------------------------------------

def test_u9_writer_creates_event_file(tmp_path):
    """U9: Tạo writer, ghi scalar, flush → event file xuất hiện dưới runs/<experiment>."""
    runs_root = tmp_path / "runs"
    writer = create_writer("exp_test", log_root=str(runs_root))
    writer.add_scalar("Loss/train", 0.5, 0)
    writer.flush()
    writer.close()

    event_files = list((runs_root / "exp_test").glob("events.out.tfevents.*"))
    assert len(event_files) >= 1, f"Không tìm thấy event file trong {(runs_root / 'exp_test')}"


# ---------------------------------------------------------------------------
# U10: Checkpoint resume round-trip + config mismatch
# ---------------------------------------------------------------------------

def test_u10_checkpoint_resume_roundtrip(tmp_path):
    """U10: Lưu/nạp resume checkpoint → khôi phục model, optimizer, epoch, score, history."""
    model = TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    history = {
        "train_loss": [0.5], "val_loss": [0.6],
        "train_acc": [0.8], "val_acc": [0.78],
        "lr": [1e-3], "epoch_time": [10.0],
    }
    phase_config = {"arch": "resnet18", "lr": 1e-3}

    ckpt_path = tmp_path / "checkpoint.pth"
    save_checkpoint(
        str(ckpt_path), model, optimizer,
        phase_epochs_completed=3, best_score=0.85, best_epoch=2,
        history=history, phase_config=phase_config,
    )

    model2 = TinyModel()
    optimizer2 = optim.SGD(model2.parameters(), lr=1e-3)
    epochs, score, best_epoch, loaded_history, es_state = load_checkpoint(
        str(ckpt_path), model2, optimizer2, device=torch.device("cpu"),
        expected_phase_config=phase_config,
    )
    assert epochs == 3
    assert score == 0.85
    assert best_epoch == 2
    assert loaded_history == history
    assert es_state is None


def test_u10_config_mismatch_refused(tmp_path):
    """U10: Resume với config phase không khớp → bị từ chối rõ ràng."""
    model = TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    history = {
        "train_loss": [0.5], "val_loss": [0.6],
        "train_acc": [0.8], "val_acc": [0.78],
        "lr": [1e-3], "epoch_time": [10.0],
    }
    ckpt_path = tmp_path / "ckpt.pth"
    save_checkpoint(
        str(ckpt_path), model, optimizer,
        phase_epochs_completed=1, best_score=0.5, best_epoch=0,
        history=history, phase_config={"arch": "resnet18"},
    )

    model2 = TinyModel()
    optimizer2 = optim.SGD(model2.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="Phase config mismatch"):
        load_checkpoint(
            str(ckpt_path), model2, optimizer2, device=torch.device("cpu"),
            expected_phase_config={"arch": "vgg16"},
        )


def test_u10_missing_checkpoint_raises(tmp_path):
    """U10: Checkpoint không tồn tại → FileNotFoundError."""
    model = TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(
            str(tmp_path / "nonexistent.pth"), model, optimizer,
            device=torch.device("cpu"), expected_phase_config={},
        )


# ---------------------------------------------------------------------------
# U11: Test metrics round-trip, no placeholder
# ---------------------------------------------------------------------------

def test_u11_metrics_roundtrip(tmp_path):
    """U11: Lưu/nạp metrics JSON → round-trip thành công."""
    metrics = {"test_loss": 0.45, "test_acc": 0.88}
    save_test_metrics(tmp_path, metrics)
    loaded = load_test_metrics(tmp_path)
    assert loaded == metrics


def test_u11_no_empty_metrics(tmp_path):
    """U11: Không tạo metrics rỗng/giả trước đánh giá."""
    with pytest.raises(ValueError):
        save_test_metrics(tmp_path, {})

