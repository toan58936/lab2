"""Kiểm thử cho going_modular.engine (Phase 5).

Chạy nhanh (không cần GPU/mạng):
    pytest tests/test_engine.py -v

Ánh xạ mã test theo spec: E1–E16.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from going_modular import engine, utils
from going_modular.engine import evaluate_step, train, train_step
from going_modular.model_builder import (
    ModelConfig,
    build_model,
    set_train_mode,
)


class CountCallsWriter:
    """TensorBoard writer giả ghi lại các lời gọi add_scalar."""

    def __init__(self):
        self.calls = []  # list of (tag, value, step)

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, value, step))

    def add_scalars(self, main_tag, tag_scalar_dict, global_step):
        for sub_tag, value in tag_scalar_dict.items():
            self.calls.append((f"{main_tag}/{sub_tag}", value, global_step))

    def flush(self):
        pass

    def close(self):
        pass


def make_toy_loader(n_samples=10, n_features=4, n_classes=3, batch_size=4, seed=0):
    """Tạo DataLoader giả với dữ liệu ngẫu nhiên cố định."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_samples, n_features, generator=g)
    y = torch.randint(0, n_classes, (n_samples,), generator=g)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size)


def make_toy_model(n_features=4, n_classes=3):
    return nn.Linear(n_features, n_classes)


# ---------------------------------------------------------------------------
# Nhóm A: Đúng đắn của một bước (E1–E4)
# ---------------------------------------------------------------------------

def test_e1_train_step_returns_float_and_trainable_only_change():
    """E1: train_step trả về float; chỉ tham số trainable thay đổi."""
    model = make_toy_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loader = make_toy_loader()

    before = {id(p): p.clone() for p in model.parameters()}
    loss, acc = train_step(model, loader, loss_fn, optimizer, torch.device("cpu"))

    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
    for p in model.parameters():
        assert not torch.equal(before[id(p)], p)


def test_e2_evaluate_step_no_grad_no_weight_change():
    """E2: evaluate_step không tạo gradient, không đổi weights, model ở eval."""
    model = make_toy_model()
    loss_fn = nn.CrossEntropyLoss()
    loader = make_toy_loader()

    before = {id(p): p.clone() for p in model.parameters()}
    model.train()
    loss, acc = evaluate_step(model, loader, loss_fn, torch.device("cpu"))

    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
    assert model.training is False
    for p in model.parameters():
        assert torch.equal(before[id(p)], p)


def test_e3_weighted_loss_uneven_batches():
    """E3: lr=0, batch cuối nhỏ hơn → loss trung bình có trọng số bằng loss toàn tập."""
    model = make_toy_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0)
    loader = make_toy_loader(n_samples=10, batch_size=4)

    loss, _ = train_step(model, loader, loss_fn, optimizer, torch.device("cpu"))
    X_all = torch.cat([b[0] for b in loader])
    y_all = torch.cat([b[1] for b in loader])
    expected = loss_fn(model(X_all), y_all).item()

    assert abs(loss - expected) < 1e-6


def test_e4_evaluate_step_known_predictions():
    """E4: Mô hình dự đoán lớp 0 → accuracy = tỷ lệ nhãn 0; trọng số không đổi; model.training == False."""
    model = nn.Linear(4, 3)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
        model.bias[0] = 10.0

    X = torch.randn(20, 4)
    y = torch.randint(0, 3, (20,))
    loader = DataLoader(TensorDataset(X, y), batch_size=5)

    before = {id(p): p.clone() for p in model.parameters()}
    model.train()
    loss, acc = evaluate_step(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))

    expected_acc = (y == 0).float().mean().item()
    assert abs(acc - expected_acc) < 1e-6
    assert model.training is False
    for p in model.parameters():
        assert torch.equal(before[id(p)], p)


# ---------------------------------------------------------------------------
# Nhóm B: BatchNorm trong huấn luyện (E5–E6)
# ---------------------------------------------------------------------------

def test_e5_train_mode_fn_called_once():
    """E5: train_mode_fn được gọi đúng 1 lần mỗi train_step."""
    model = make_toy_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loader = make_toy_loader()

    call_count = 0

    def counting_fn(m):
        nonlocal call_count
        call_count += 1
        m.train()

    train_step(model, loader, loss_fn, optimizer, torch.device("cpu"), train_mode_fn=counting_fn)
    assert call_count == 1

    train_step(model, loader, loss_fn, optimizer, torch.device("cpu"))
    assert model.training is True


def test_e6_bn_frozen_running_mean_unchanged():
    """E6: ResNet-18 Giai đoạn 1, set_train_mode làm train_mode_fn → bn1.running_mean không đổi."""
    cfg = ModelConfig(arch="resnet18", weights=None)
    model = build_model(cfg)
    X = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 10, (4,))
    ds = TensorDataset(X, y)
    loader_big = DataLoader(ds, batch_size=2)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    before = model.bn1.running_mean.clone()
    train_step(model, loader_big, loss_fn, optimizer, torch.device("cpu"),
               train_mode_fn=lambda m: set_train_mode(m, cfg))

    assert model.training is True
    assert torch.equal(before, model.bn1.running_mean), "BN đóng băng không được thay đổi running_mean"


# ---------------------------------------------------------------------------
# Nhóm C: Vòng lặp nhiều epoch (E7–E10)
# ---------------------------------------------------------------------------

def test_e7_history_six_keys_correct_length():
    """E7: train(epochs=3) → history có 6 khóa, mỗi danh sách dài 3."""
    model = make_toy_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    train_dl = make_toy_loader(n_samples=20, batch_size=5)
    val_dl = make_toy_loader(n_samples=10, batch_size=5, seed=1)

    history = train(model, train_dl, val_dl, optimizer, loss_fn,
                    epochs=3, device=torch.device("cpu"), verbose=False)

    assert set(history.keys()) == {
        "train_loss", "val_loss", "train_acc", "val_acc", "lr", "epoch_time"
    }
    for key in history:
        assert len(history[key]) == 3, f"Khóa {key} có độ dài {len(history[key])}"
    assert isinstance(history["lr"][0], list)


def test_e8_best_model_saved_at_best_epoch(tmp_path):
    """E8: val_acc scripted [0.5, 0.9, 0.7] → best_model.pth là trạng thái epoch 2 (index 1)."""
    model = make_toy_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0)  # trọng số không đổi giữa các epoch
    train_dl = make_toy_loader(n_samples=20, batch_size=5)
    val_dl = make_toy_loader(n_samples=10, batch_size=5, seed=1)

    val_accs = iter([0.5, 0.9, 0.7])

    def fake_eval(m, dl, lf, dev):
        return 0.1, next(val_accs)

    engine.train_step = train_step
    engine.evaluate_step = fake_eval
    try:
        best_path = os.path.join(tmp_path, "best.pth")
        initial_w = model.weight.clone()

        train(model, train_dl, val_dl, optimizer, loss_fn,
              epochs=3, device=torch.device("cpu"), save_path=best_path,
              monitor="val_acc", verbose=False)
    finally:
        engine.train_step = train_step
        engine.evaluate_step = evaluate_step

    saved = torch.load(best_path, weights_only=True)
    # val_acc cao nhất là 0.9 ở epoch index 1 (epoch 2). Với lr=0, trọng số không đổi,
    # nên saved trọng số phải bằng initial_w.
    assert torch.equal(saved["weight"], initial_w)


def test_e9_restore_best_true_false(tmp_path):
    """E9: restore_best=True → trọng số bằng epoch tốt nhất; False → epoch cuối."""
    best_path = os.path.join(tmp_path, "best.pth")

    # Chạy với restore_best=True (mặc định)
    model1 = make_toy_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(model1.parameters(), lr=0.0)
    train_dl = make_toy_loader(n_samples=20, batch_size=5)
    val_dl = make_toy_loader(n_samples=10, batch_size=5, seed=1)

    val_accs = iter([0.5, 0.9, 0.7])

    def fake_eval(m, dl, lf, dev):
        return 0.1, next(val_accs)

    engine.train_step = train_step
    engine.evaluate_step = fake_eval
    try:
        train(model1, train_dl, val_dl, optimizer1, loss_fn,
              epochs=3, device=torch.device("cpu"), save_path=best_path,
              restore_best=True, verbose=False)
    finally:
        engine.train_step = train_step
        engine.evaluate_step = evaluate_step

    w_restore_best = model1.weight.clone()
    initial1 = model1.weight.clone()

    # Chạy với restore_best=False
    val_accs2 = iter([0.5, 0.9, 0.7])
    engine.evaluate_step = lambda m, dl, lf, dev: (0.1, next(val_accs2))
    try:
        model2 = make_toy_model()
        initial2 = model2.weight.clone()
        optimizer2 = optim.SGD(model2.parameters(), lr=0.0)
        train(model2, train_dl, val_dl, optimizer2, loss_fn,
              epochs=3, device=torch.device("cpu"), save_path=best_path,
              restore_best=False, verbose=False)
    finally:
        engine.train_step = train_step
        engine.evaluate_step = evaluate_step

    w_no_restore = model2.weight.clone()

    # Với lr=0, trọng số không đổi trong cả hai lần → mỗi model giữ nguyên trọng số ban đầu
    assert torch.equal(w_restore_best, initial1)
    assert torch.equal(w_no_restore, initial2)


def test_e10_invalid_args():
    """E10: epochs=0, start_epoch=-1, monitor='foo' → ValueError."""
    model = make_toy_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    train_dl = make_toy_loader()
    val_dl = make_toy_loader(seed=1)

    with pytest.raises(ValueError):
        train(model, train_dl, val_dl, optimizer, loss_fn, epochs=0,
              device=torch.device("cpu"), verbose=False)

    with pytest.raises(ValueError):
        train(model, train_dl, val_dl, optimizer, loss_fn, epochs=3,
              device=torch.device("cpu"), start_epoch=-1, verbose=False)

    with pytest.raises(ValueError):
        train(model, train_dl, val_dl, optimizer, loss_fn, epochs=3,
              device=torch.device("cpu"), monitor="foo", verbose=False)


# ---------------------------------------------------------------------------
# Nhóm D: TensorBoard và nối tiếp giai đoạn (E11–E12)
# ---------------------------------------------------------------------------

def test_e11_writer_tags_and_steps():
    """E11: writer giả → bước ghi 6 và 7 (epochs=2, start_epoch=5); đủ 5 tag."""
    model = make_toy_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    train_dl = make_toy_loader(n_samples=20, batch_size=5)
    val_dl = make_toy_loader(n_samples=10, batch_size=5, seed=1)

    writer = CountCallsWriter()
    train(model, train_dl, val_dl, optimizer, loss_fn,
          epochs=2, device=torch.device("cpu"), start_epoch=5,
          writer=writer, verbose=False)

    steps = {step for _, _, step in writer.calls}
    assert steps == {6, 7}

    tags_at_step_6 = {tag for tag, _, step in writer.calls if step == 6}
    expected_tags = {"Loss/train", "Loss/val", "Accuracy/train", "Accuracy/val", "LR/group_0"}
    assert tags_at_step_6 == expected_tags


def test_e12_writer_none_is_noop():
    """E12: train với writer=None → chạy bình thường, không lỗi."""
    model = make_toy_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    train_dl = make_toy_loader(n_samples=20, batch_size=5)
    val_dl = make_toy_loader(n_samples=10, batch_size=5, seed=1)

    history = train(model, train_dl, val_dl, optimizer, loss_fn,
                    epochs=2, device=torch.device("cpu"), writer=None, verbose=False)
    assert len(history["train_loss"]) == 2


# ---------------------------------------------------------------------------
# E13–E16: Resume, atomic interruption, monitor, timing
# ---------------------------------------------------------------------------

def test_e13_resume_matching_config(tmp_path):
    """E13: Resume khớp config → khôi phục epoch mới nhất, nối history từ epoch tiếp theo."""
    model = make_toy_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    train_dl = make_toy_loader(n_samples=20, batch_size=5)
    val_dl = make_toy_loader(n_samples=10, batch_size=5, seed=1)

    ckpt_dir = tmp_path / "artifacts"
    ckpt_dir.mkdir()

    history_phase1 = train(model, train_dl, val_dl, optimizer, loss_fn,
                           epochs=2, device=torch.device("cpu"),
                           checkpoint_dir=str(ckpt_dir), verbose=False)
    assert len(history_phase1["train_loss"]) == 2

    model2 = make_toy_model()
    optimizer2 = optim.SGD(model2.parameters(), lr=1e-3)
    history_phase2 = train(model2, train_dl, val_dl, optimizer2, loss_fn,
                           epochs=4, device=torch.device("cpu"),
                           checkpoint_dir=str(ckpt_dir), resume=True, verbose=False)

    assert len(history_phase2["train_loss"]) == 4
    assert history_phase2["train_loss"][:2] == history_phase1["train_loss"]


def test_e13_resume_config_mismatch_rejected(tmp_path):
    """E13: Resume config không khớp → ValueError, không ghi đè artifacts."""
    model = make_toy_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    train_dl = make_toy_loader(n_samples=20, batch_size=5)
    val_dl = make_toy_loader(n_samples=10, batch_size=5, seed=1)

    ckpt_dir = tmp_path / "artifacts_mismatch"
    ckpt_dir.mkdir()

    train(model, train_dl, val_dl, optimizer, loss_fn,
          epochs=2, device=torch.device("cpu"),
          checkpoint_dir=str(ckpt_dir), verbose=False)

    model2 = make_toy_model()
    optimizer2 = optim.SGD(model2.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="Phase config mismatch"):
        train(model2, train_dl, val_dl, optimizer2, loss_fn,
              epochs=4, device=torch.device("cpu"),
              checkpoint_dir=str(ckpt_dir), resume=True,
              expected_phase_config={"arch": "vgg16"}, verbose=False)


def test_e14_atomic_interruption_checkpoint(tmp_path, monkeypatch):
    """E14: os.replace lỗi khi lưu checkpoint → file cũ nguyên vẹn, temp dọn, lỗi truyền lên."""
    model = make_toy_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    train_dl = make_toy_loader(n_samples=20, batch_size=5)
    val_dl = make_toy_loader(n_samples=10, batch_size=5, seed=1)

    ckpt_dir = tmp_path / "artifacts_atomic"
    ckpt_dir.mkdir()

    train(model, train_dl, val_dl, optimizer, loss_fn,
          epochs=1, device=torch.device("cpu"),
          checkpoint_dir=str(ckpt_dir), verbose=False)
    ckpt_path = ckpt_dir / "last_checkpoint.pth"
    assert ckpt_path.is_file()
    original_bytes = ckpt_path.read_bytes()

    # Inject os.replace failure during real save_checkpoint
    original_replace = os.replace

    def failing_replace(src, dst):
        raise OSError("disk full on replace")

    monkeypatch.setattr(os, "replace", failing_replace)

    shutil.copytree(ckpt_dir, tmp_path / "artifacts_atomic_copy")
    ckpt_dir2 = tmp_path / "artifacts_atomic_copy"

    model2 = make_toy_model()
    optimizer2 = optim.SGD(model2.parameters(), lr=1e-3)
    with pytest.raises(OSError, match="disk full on replace"):
        train(model2, train_dl, val_dl, optimizer2, loss_fn,
              epochs=3, device=torch.device("cpu"),
              checkpoint_dir=str(ckpt_dir2), resume=True, verbose=False)

    assert (ckpt_dir2 / "last_checkpoint.pth").read_bytes() == original_bytes
    tmp_files = list(ckpt_dir2.glob("*.tmp"))
    assert len(tmp_files) == 0, f"Temp files còn lại: {tmp_files}"


def test_e15_monitor_val_loss_selects_min(tmp_path):
    """E15: monitor='val_loss' → chọn epoch có loss nhỏ nhất, restore_best khôi phục trọng số đó."""
    model = make_toy_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0)
    train_dl = make_toy_loader(n_samples=20, batch_size=5)
    val_dl = make_toy_loader(n_samples=10, batch_size=5, seed=1)

    best_path = os.path.join(tmp_path, "best.pth")

    val_losses = iter([0.9, 0.4, 0.7])

    def fake_eval(m, dl, lf, dev):
        return next(val_losses), 0.5

    engine.train_step = train_step
    engine.evaluate_step = fake_eval
    try:
        initial_w = model.weight.clone()
        train(model, train_dl, val_dl, optimizer, loss_fn,
              epochs=3, device=torch.device("cpu"), monitor="val_loss",
              save_path=best_path, restore_best=True, verbose=False)
    finally:
        engine.train_step = train_step
        engine.evaluate_step = evaluate_step

    saved = torch.load(best_path, weights_only=True)
    assert torch.equal(saved["weight"], initial_w)


def test_e16_timing_and_history_lr():
    """E16: epoch_time được ghi một lần mỗi epoch; lr phản ánh tất cả param group."""
    model = make_toy_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    train_dl = make_toy_loader(n_samples=20, batch_size=5)
    val_dl = make_toy_loader(n_samples=10, batch_size=5, seed=1)

    history = train(model, train_dl, val_dl, optimizer, loss_fn,
                    epochs=2, device=torch.device("cpu"), verbose=False)

    assert len(history["epoch_time"]) == 2
    assert all(isinstance(t, float) and t >= 0 for t in history["epoch_time"])
    assert len(history["lr"]) == 2
    assert all(isinstance(lr_list, list) for lr_list in history["lr"])
    assert all(len(lr_list) == 1 for lr_list in history["lr"])


# ---------------------------------------------------------------------------
# New API: save_path and train_mode_fn
# ---------------------------------------------------------------------------

def test_save_path_saves_best_model(tmp_path):
    """save_path (API mới) lưu best model checkpoint."""
    model = make_toy_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    train_dl = make_toy_loader(n_samples=20, batch_size=5)
    val_dl = make_toy_loader(n_samples=10, batch_size=5, seed=1)

    ckpt_path = os.path.join(tmp_path, "best_new.pth")

    train(model, train_dl, val_dl, optimizer, loss_fn,
          epochs=2, device=torch.device("cpu"),
          save_path=ckpt_path, verbose=False)

    assert os.path.isfile(ckpt_path)


def test_train_mode_fn_keeps_backbone_eval(tmp_path):
    """train_mode_fn (API mới) có thể giữ backbone ở eval trong quá trình train."""
    model = make_toy_model()
    backbone = nn.Linear(4, 4)
    for p in backbone.parameters():
        p.requires_grad = False

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    train_dl = make_toy_loader(n_samples=20, batch_size=5)
    val_dl = make_toy_loader(n_samples=10, batch_size=5, seed=1)

    train(model, train_dl, val_dl, optimizer, loss_fn,
          epochs=1, device=torch.device("cpu"),
          train_mode_fn=lambda m: setattr(m, 'training', True) or backbone.eval(),
          verbose=False)

    assert backbone.training is False
