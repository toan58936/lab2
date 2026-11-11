"""Kiểm thử cho going_modular.model_builder (Phase 2).

Chạy nhanh (không tải ImageNet weights):
    pytest tests/test_model_builder.py -v -m "not slow"

Chạy toàn bộ (bao gồm slow S1–S2, cần mạng/ImageNet):
    pytest tests/test_model_builder.py -v

Ánh xạ mã test: T1–T14c (nhanh), S1–S2 (chậm, đánh dấu slow).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from unittest.mock import patch

from going_modular.model_builder import (
    ModelConfig,
    ARCH_SPECS,
    build_model,
    count_params,
    MyModel,
    get_param_groups,
    set_train_mode,
)

ARCHITECTURES = ["resnet18", "vgg16"]


# ---------------------------------------------------------------------------
# Nhóm A: Cấu trúc và hình dạng
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_t1_output_shape(arch):
    """T1: Đầu ra có shape (2, 10) với đầu vào (2, 3, 224, 224)."""
    model = build_model(ModelConfig(arch=arch, weights=None))
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 10)


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_t2_head_replaced(arch):
    """T2: Head là nn.Linear, out_features == num_classes (7 != 10)."""
    model = build_model(ModelConfig(arch=arch, num_classes=7, weights=None))
    head = ARCH_SPECS[arch]["get_head"](model)
    assert isinstance(head, nn.Linear)
    assert head.out_features == 7


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_t2b_head_with_dropout(arch):
    """T2b: dropout=0.3 → head là Sequential(Dropout(0.3), Linear(out=10))."""
    model = build_model(ModelConfig(arch=arch, dropout=0.3, weights=None))
    head = ARCH_SPECS[arch]["get_head"](model)
    assert isinstance(head, nn.Sequential)
    assert isinstance(head[0], nn.Dropout)
    assert head[0].p == 0.3
    assert isinstance(head[1], nn.Linear)
    assert head[1].out_features == 10


# ---------------------------------------------------------------------------
# Nhóm B: Trạng thái đóng băng
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_t3_only_head_trainable(arch):
    """T3: Chỉ tham số head trainable; phần còn lại frozen."""
    model = build_model(ModelConfig(arch=arch, weights=None))
    head = ARCH_SPECS[arch]["get_head"](model)
    head_ids = {id(p) for p in head.parameters()}

    for p in model.parameters():
        if id(p) in head_ids:
            assert p.requires_grad, f"Tham số head phải trainable (arch={arch})"
        else:
            assert not p.requires_grad, f"Tham số backbone phải frozen (arch={arch})"


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_t4_param_counts(arch):
    """T4: Số tham số khớp giá trị kỳ erwartet ở Giai đoạn 1."""
    expected = {
        "resnet18": {"total": 11_181_642, "trainable": 5_130},
        "vgg16": {"total": 14_965_578, "trainable": 250_890},
    }[arch]

    model = build_model(ModelConfig(arch=arch, weights=None))
    stats = count_params(model)

    assert stats["total"] == expected["total"], f"Tổng sai cho {arch}"
    assert stats["trainable"] == expected["trainable"], f"Trainable sai cho {arch}"
    assert stats["frozen"] == expected["total"] - expected["trainable"]


@pytest.mark.parametrize("arch", ARCHITECTURES)
@pytest.mark.parametrize("n", [1, 2])
def test_t5_unfreeze_last_n(arch, n):
    """T5: n khối cuối trainable; các khối trước đó frozen."""
    cfg = ModelConfig(arch=arch, unfreeze_last_n=n, weights=None)
    model = build_model(cfg)
    blocks = ARCH_SPECS[arch]["blocks"](model)

    for block in blocks[-n:]:
        for p in block.parameters():
            assert p.requires_grad, f"Khối cuối {block} phải trainable"

    for block in blocks[:-n] if n < len(blocks) else []:
        for p in block.parameters():
            assert not p.requires_grad, f"Khối trước {block} phải frozen"


def test_t5b_unfreeze_increases_trainable():
    """T5b: unfreeze_last_n=1 có nhiều tham số trainable hơn unfreeze_last_n=0."""
    base = count_params(
        build_model(ModelConfig(arch="resnet18", unfreeze_last_n=0, weights=None))
    )["trainable"]
    unfrozen = count_params(
        build_model(ModelConfig(arch="resnet18", unfreeze_last_n=1, weights=None))
    )["trainable"]
    assert unfrozen > base


def test_t5c_full_training_mode_all_trainable():
    """T5c: freeze_backbone=False → mọi tham số trainable."""
    model = build_model(ModelConfig(arch="resnet18", freeze_backbone=False, weights=None))
    stats = count_params(model)
    assert stats["trainable"] == stats["total"]


# ---------------------------------------------------------------------------
# Nhóm C: Kiểm tra cấu hình
# ---------------------------------------------------------------------------

def test_t6_unfreeze_more_than_blocks():
    """T6: unfreeze_last_n vượt quá số khối → ValueError."""
    with pytest.raises(ValueError):
        build_model(ModelConfig(arch="resnet18", unfreeze_last_n=5, weights=None))
    with pytest.raises(ValueError):
        build_model(ModelConfig(arch="vgg16", unfreeze_last_n=6, weights=None))


def test_t7_unfreeze_without_freeze():
    """T7: freeze_backbone=False + unfreeze_last_n>0 → ValueError."""
    with pytest.raises(ValueError):
        build_model(
            ModelConfig(arch="resnet18", freeze_backbone=False, unfreeze_last_n=1, weights=None)
        )


def test_t8_unknown_arch():
    """T8: kiến trúc không xác định → ValueError."""
    with pytest.raises(ValueError):
        build_model(ModelConfig(arch="not_a_model", weights=None))


def test_t8b_invalid_values():
    """T8b: unfreeze_last_n<0 và dropout=1.0 → ValueError."""
    with pytest.raises(ValueError):
        build_model(ModelConfig(arch="resnet18", unfreeze_last_n=-1, weights=None))
    with pytest.raises(ValueError):
        build_model(ModelConfig(arch="resnet18", dropout=1.0, weights=None))


def test_t8c_validate_no_model_construction_on_invalid_arch():
    """T8c: invalid arch raises ValueError BEFORE any model is constructed."""
    with patch("going_modular.model_builder._resolve_weights") as mock_resolve:
        mock_resolve.side_effect = AssertionError("_resolve_weights should not be called during validation")
        with pytest.raises(ValueError, match="Kiến trúc không hỗ trợ"):
            build_model(ModelConfig(arch="not_a_model", weights=None))
        mock_resolve.assert_not_called()


def test_t8d_validate_no_model_construction_on_unfreeze_exceeds():
    """T8d: unfreeze_last_n > num_blocks raises ValueError BEFORE any model is constructed."""
    with patch("going_modular.model_builder._resolve_weights") as mock_resolve:
        mock_resolve.side_effect = AssertionError("_resolve_weights should not be called during validation")
        with pytest.raises(ValueError, match="vượt quá số khối"):
            build_model(ModelConfig(arch="resnet18", unfreeze_last_n=5, weights=None))
        mock_resolve.assert_not_called()


def test_t8e_validate_no_model_construction_on_invalid_values():
    """T8e: unfreeze_last_n<0 and dropout>=1 raise ValueError BEFORE any model is constructed."""
    with patch("going_modular.model_builder._resolve_weights") as mock_resolve:
        mock_resolve.side_effect = AssertionError("_resolve_weights should not be called during validation")
        with pytest.raises(ValueError, match="không được âm"):
            build_model(ModelConfig(arch="resnet18", unfreeze_last_n=-1, weights=None))
        with pytest.raises(ValueError, match="phải nằm trong"):
            build_model(ModelConfig(arch="resnet18", dropout=1.0, weights=None))
        mock_resolve.assert_not_called()


# ---------------------------------------------------------------------------
# Nhóm D: BatchNorm và optimizer
# ---------------------------------------------------------------------------

def test_t9_bn_eval_when_fully_frozen():
    """T9: set_train_mode → model.training=True, mọi BN training=False."""
    model = build_model(ModelConfig(arch="resnet18", weights=None))
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert bn_modules, "Phải có ít nhất một BatchNorm2d"

    set_train_mode(model, ModelConfig(arch="resnet18", freeze_bn_stats=True))

    assert model.training is True
    for bn in bn_modules:
        assert bn.training is False


def test_t10_bn_train_only_for_unfrozen_blocks():
    """T10: unfreeze_last_n=1 → BN layer4 train, BN bn1/layer1-3 eval."""
    cfg = ModelConfig(arch="resnet18", unfreeze_last_n=1, weights=None)
    model = build_model(cfg)
    set_train_mode(model, cfg)

    assert model.backbone.bn1.training is False
    assert model.backbone.layer1[0].bn1.training is False
    assert model.backbone.layer2[0].bn1.training is False
    assert model.backbone.layer3[0].bn1.training is False
    assert model.backbone.layer4[0].bn1.training is True


def test_t10b_freeze_bn_stats_disabled():
    """T10b: freeze_bn_stats=False → mọi BN ở chế độ train."""
    cfg = ModelConfig(arch="resnet18", freeze_bn_stats=False, weights=None)
    model = build_model(cfg)
    set_train_mode(model, cfg)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            assert m.training is True


def test_t10c_running_stats_unchanged_when_frozen():
    """T10c: running_mean của BN đóng băng không thay đổi sau forward train-mode."""
    cfg = ModelConfig(arch="resnet18", weights=None)
    model = build_model(cfg)
    before = model.backbone.bn1.running_mean.clone()

    set_train_mode(model, cfg)
    model(torch.randn(4, 3, 64, 64))

    assert torch.equal(before, model.backbone.bn1.running_mean)


def test_t11_groups_stage1():
    """T11: Giai đoạn 1 → một nhóm, lr=1e-3, không nhóm rỗng."""
    model = build_model(ModelConfig(arch="resnet18", weights=None))
    cfg = ModelConfig(arch="resnet18", weights=None)
    groups = get_param_groups(model, cfg, lr_head=1e-3, lr_backbone=1e-4)

    assert len(groups) == 1
    assert groups[0]["lr"] == 1e-3

    head = ARCH_SPECS["resnet18"]["get_head"](model)
    head_ids = {id(p) for p in head.parameters()}
    group_ids = {id(p) for g in groups for p in g["params"]}
    assert head_ids == group_ids


def test_t11_groups_stage2():
    """T11: Giai đoạn 2 → hai nhóm, lr=[1e-3, 1e-4], không trùng tham số."""
    model = build_model(ModelConfig(arch="resnet18", unfreeze_last_n=1, weights=None))
    cfg = ModelConfig(arch="resnet18", unfreeze_last_n=1, weights=None)
    groups = get_param_groups(model, cfg, lr_head=1e-3, lr_backbone=1e-4)

    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-3
    assert groups[1]["lr"] == 1e-4

    all_ids = [id(p) for g in groups for p in g["params"]]
    assert len(all_ids) == len(set(all_ids)), "Không được trùng tham số giữa các nhóm"


# ---------------------------------------------------------------------------
# Nhóm E: Tái lập, lưu/nạp cấu hình, tiền xử lý
# ---------------------------------------------------------------------------

def test_t12_reproducible_with_same_seed():
    """T12: Cùng seed → cùng state_dict."""
    torch.manual_seed(42)
    m1 = build_model(ModelConfig(arch="resnet18", weights=None))
    sd1 = m1.state_dict()

    torch.manual_seed(42)
    m2 = build_model(ModelConfig(arch="resnet18", weights=None))
    sd2 = m2.state_dict()

    assert sd1.keys() == sd2.keys()
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k])


def test_t13_config_roundtrip():
    """T13: from_dict(to_dict()) giữ nguyên cấu hình."""
    cfg = ModelConfig(arch="vgg16", num_classes=7, dropout=0.2, unfreeze_last_n=2)
    restored = ModelConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_t13b_unknown_key_rejected():
    """T13b: from_dict với khóa lạ → TypeError."""
    with pytest.raises(TypeError):
        ModelConfig.from_dict({"arch": "resnet18", "foo": 1})


def test_t13c_config_is_frozen():
    """T13c: ModelConfig là frozen → gán thuộc tính báo lỗi."""
    cfg = ModelConfig()
    with pytest.raises(Exception):
        cfg.arch = "vgg16"


# ---------------------------------------------------------------------------
# Nhóm F: Trọng số ImageNet thật (chậm, đánh dấu slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_slow_pretrained_only_head_trainable():
    """S1: ResNet-18 với IMAGENET1K_V1 → chỉ head trainable, trainable=5130."""
    try:
        model = build_model(ModelConfig(arch="resnet18", weights="IMAGENET1K_V1"))
    except OSError:
        pytest.skip("Không tải được ImageNet weights (mạng/môi trường).")

    head = model.head
    head_ids = {id(p) for p in head.parameters()}
    for p in model.parameters():
        if id(p) in head_ids:
            assert p.requires_grad
        else:
            assert not p.requires_grad

    stats = count_params(model)
    assert stats["trainable"] == 5_130


@pytest.mark.slow
def test_slow_pretrained_backbone_differs_from_random():
    """S2: conv1.weight pre-trained khác với khởi tạo ngẫu nhiên."""
    try:
        m_pretrained = build_model(ModelConfig(arch="resnet18", weights="IMAGENET1K_V1"))
        m_random = build_model(ModelConfig(arch="resnet18", weights=None))
    except OSError:
        pytest.skip("Không tải được ImageNet weights (mạng/môi trường).")

    assert not torch.equal(
        m_pretrained.backbone.conv1.weight, m_random.backbone.conv1.weight
    )



