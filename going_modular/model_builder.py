
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights, VGG16_Weights


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    arch: str = "resnet18"
    num_classes: int = 10
    weights: str | None = "IMAGENET1K_V1"
    freeze_backbone: bool = True
    unfreeze_last_n: int = 0
    dropout: float = 0.0
    freeze_bn_stats: bool = True

    def to_dict(self) -> dict:
        """Chuyển config thành dict (phục vụ lưu config.json)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Tạo ModelConfig từ dict; từ chối khóa không xác định bằng TypeError."""
        valid_names = {f.name for f in fields(cls)}
        unknown = set(d) - valid_names
        if unknown:
            raise TypeError(
                f"Khóa không được định nghĩa trong ModelConfig: {sorted(unknown)}"
            )
        return cls(**d)


# ---------------------------------------------------------------------------
# Architecture helpers / registry (native torchvision models)
# ---------------------------------------------------------------------------

def _vgg16_conv_blocks(model: nn.Module) -> list[nn.Sequential]:
    """Chia VGG16 features thành 5 khối conv, mỗi khối kết thúc bằng MaxPool2d."""
    blocks: list[nn.Sequential] = []
    current: list[nn.Module] = []
    for module in model.features:
        current.append(module)
        if isinstance(module, nn.MaxPool2d):
            blocks.append(nn.Sequential(*current))
            current = []
    return blocks


def _head_param_ids(model: nn.Module, cfg: ModelConfig) -> set[int]:
    """Trả về id tham số của head native (fc hoặc classifier[6])."""
    if hasattr(model, "fc"):
        head = model.fc
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential) and len(model.classifier) > 6:
        head = model.classifier[6]
    elif hasattr(model, "classifier"):
        head = model.classifier
    else:
        raise AttributeError("Không xác định được head của mô hình.")
    return {id(p) for p in head.parameters()}


ARCH_SPECS: dict[str, dict] = {
    "resnet18": {
        "blocks": lambda m: [m.layer1, m.layer2, m.layer3, m.layer4],
        "num_blocks": 4,
    },
    "vgg16": {
        "blocks": _vgg16_conv_blocks,
        "num_blocks": 5,
    },
}

_WEIGHT_ENUMS: dict[str, type] = {
    "resnet18": ResNet18_Weights,
    "vgg16": VGG16_Weights,
}


def _resolve_weights(arch: str, weights_name: str | None):
    """Ánh xạ tên trọng số (str) thành torchvision WeightsEnum member."""
    if weights_name is None:
        return None
    enum_cls = _WEIGHT_ENUMS.get(arch)
    if enum_cls is None:
        raise ValueError(f"Không hỗ trợ kiến trúc '{arch}' để giải quyết trọng số.")
    try:
        return enum_cls[weights_name]
    except (KeyError, AttributeError) as exc:
        raise ValueError(
            f"Không xác định được trọng số '{weights_name}' cho kiến trúc '{arch}'."
        ) from exc


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(cfg: ModelConfig) -> None:
    if cfg.arch not in ARCH_SPECS:
        raise ValueError(
            f"Kiến trúc không hỗ trợ: '{cfg.arch}'. "
            f"Các kiến trúc hợp lệ: {sorted(ARCH_SPECS)}."
        )
    if cfg.unfreeze_last_n < 0:
        raise ValueError(f"unfreeze_last_n không được âm (đã nhận {cfg.unfreeze_last_n}).")
    if not (0.0 <= cfg.dropout < 1.0):
        raise ValueError(
            f"dropout phải nằm trong [0, 1) (đã nhận {cfg.dropout})."
        )
    num_blocks = ARCH_SPECS[cfg.arch]["num_blocks"]
    if cfg.unfreeze_last_n > num_blocks:
        raise ValueError(
            f"unfreeze_last_n={cfg.unfreeze_last_n} vượt quá số khối của "
            f"'{cfg.arch}' ({num_blocks})."
        )
    if not cfg.freeze_backbone and cfg.unfreeze_last_n > 0:
        raise ValueError(
            "Không thể kết hợp freeze_backbone=False với unfreeze_last_n>0. "
            "Dùng freeze_backbone=True, unfreeze_last_n=N cho Fine-Tuning."
        )


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

def _create_native_model(arch: str, weights_name: str | None) -> nn.Module:
    resolved = _resolve_weights(arch, weights_name)
    if arch == "resnet18":
        return models.resnet18(weights=resolved)
    if arch == "vgg16":
        return models.vgg16(weights=resolved)
    raise ValueError(f"Kiến trúc không hỗ trợ: '{arch}'.")


def _replace_head(model: nn.Module, cfg: ModelConfig) -> None:
    if cfg.arch == "resnet18":
        in_features = model.fc.in_features
        if cfg.dropout > 0:
            model.fc = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(in_features, cfg.num_classes))
        else:
            model.fc = nn.Linear(in_features, cfg.num_classes)
        return

    if cfg.arch == "vgg16":
        # G1 (native, sát torchvision): giữ features/avgpool/classifier[0..5], chỉ thay classifier[6]
        in_features = model.classifier[6].in_features
        if cfg.dropout > 0:
            model.classifier[6] = nn.Sequential(
                nn.Dropout(cfg.dropout), nn.Linear(in_features, cfg.num_classes)
            )
        else:
            model.classifier[6] = nn.Linear(in_features, cfg.num_classes)
        return

    raise ValueError(f"Kiến trúc không hỗ trợ: '{cfg.arch}'.")


def build_model(cfg: ModelConfig) -> nn.Module:

    _validate(cfg)

    model = _create_native_model(cfg.arch, cfg.weights)

    # Đóng băng toàn bộ trước khi thay head để head mới luôn trainable.
    if cfg.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    _replace_head(model, cfg)

    # Mở khóa N khối cuối backbone (Fine-Tuning).
    if cfg.unfreeze_last_n > 0:
        spec = ARCH_SPECS[cfg.arch]
        blocks = spec["blocks"](model)
        for block in blocks[-cfg.unfreeze_last_n:]:
            for param in block.parameters():
                param.requires_grad = True

    return model


# ---------------------------------------------------------------------------
# Train mode (BatchNorm handling)
# ---------------------------------------------------------------------------

def set_train_mode(model: nn.Module, cfg: ModelConfig) -> None:
    model.train()
    if cfg.freeze_bn_stats:
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Nếu không có tham số affine/trainable, giữ eval để không cập nhật running stats
                params = list(module.parameters(recurse=False))
                if not params or all(not p.requires_grad for p in params):
                    module.eval()


# ---------------------------------------------------------------------------
# Parameter groups for optimizer
# ---------------------------------------------------------------------------

def get_param_groups(
    model: nn.Module,
    cfg: ModelConfig,
    lr_head: float,
    lr_backbone: float,
) -> list[dict]:
    """Tạo nhóm tham số cho optimizer với LR riêng cho head và backbone.

    Group 1: tham số trainable của head (lr_head).
    Group 2 (chỉ khi có backbone trainable): tham số trainable còn lại (lr_backbone).
    Không tạo nhóm rỗng, không trùng, không bỏ sót.
    """
    head_ids = _head_param_ids(model, cfg)

    head_params = []
    backbone_params = []

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head_params.append(p)
        else:
            backbone_params.append(p)

    groups: list[dict] = []
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})

    return groups


# ---------------------------------------------------------------------------
# Parameter statistics
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> dict:
    """Trả về {total, trainable, frozen} cho mô hình."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
