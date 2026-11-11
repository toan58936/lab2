
from __future__ import annotations

from dataclasses import dataclass, fields
import torch
from torch import nn
from torchvision import models
from torchvision.models import VGG16_Weights, ResNet18_Weights


# ---------------------------------------------------------------------------
# Wrapper model: backbone torchvision + head tùy chỉnh
# ---------------------------------------------------------------------------

class MyModel(nn.Module):


    def __init__(
        self,
        arch: str,
        num_classes: int,
        weights: str | None = "IMAGENET1K_V1",
        freeze_backbone: bool = True,
        dropout: float = 0.0,
        freeze_bn_stats: bool = True,
    ):
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.freeze_bn_stats = freeze_bn_stats

        # --- Nạp backbone torchvision ---
        if arch == "resnet18":
            backbone = models.resnet18(weights=_resolve_weights(arch, weights))
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()  # bỏ lớp gốc 1000 lớp
        elif arch == "vgg16":
            backbone = models.vgg16(weights=_resolve_weights(arch, weights))
            # Tính output dim của backbone sau features+avgpool+flatten (trước classifier)
            with torch.no_grad():
                _dummy = torch.randn(1, 3, 224, 224)
                _feat = backbone.features(_dummy)
                _feat = backbone.avgpool(_feat)
                _feat = torch.flatten(_feat, 1)
                in_features = int(_feat.shape[1])
            backbone.classifier = nn.Identity()  # bỏ lớp gốc 1000 lớp
        else:
            raise ValueError(f"Kiến trúc không hỗ trợ: '{arch}'.")

        self.backbone = backbone

        # --- Head mới (luôn trainable) ---
        if dropout > 0:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.head = nn.Linear(in_features, num_classes)

        # --- Đóng băng backbone nếu yêu cầu ---
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


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
# Architecture registry (dành cho get_param_groups / unfreeze_last_n)
# ---------------------------------------------------------------------------

def _vgg16_conv_blocks(model: nn.Module) -> list[nn.Sequential]:
    """Chia VGG16 backbone.features thành 5 khối conv, mỗi khối kết thúc bằng MaxPool2d."""
    blocks: list[nn.Sequential] = []
    current: list[nn.Module] = []
    for module in model.backbone.features:
        current.append(module)
        if isinstance(module, nn.MaxPool2d):
            blocks.append(nn.Sequential(*current))
            current = []
    return blocks


ARCH_SPECS: dict[str, dict] = {
    "resnet18": {
        "get_head": lambda m: m.head,
        "set_head": lambda m, h: setattr(m, "head", h),
        "blocks": lambda m: [m.backbone.layer1, m.backbone.layer2, m.backbone.layer3, m.backbone.layer4],
        "num_blocks": 4,
    },
    "vgg16": {
        "get_head": lambda m: m.head,
        "set_head": lambda m, h: setattr(m, "head", h),
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
# build_model (new API)
# ---------------------------------------------------------------------------

def build_model(cfg: ModelConfig) -> nn.Module:
    """Xây dựng MyModel từ ModelConfig (wrapper quanh torchvision backbone).

    Thứ tự: validate → nạp backbone → thay head → đóng băng backbone → unfreeze N khối cuối.
    """
    _validate(cfg)

    model = MyModel(
        arch=cfg.arch,
        num_classes=cfg.num_classes,
        weights=cfg.weights,
        freeze_backbone=cfg.freeze_backbone,
        dropout=cfg.dropout,
        freeze_bn_stats=cfg.freeze_bn_stats,
    )

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
            if isinstance(module, nn.BatchNorm2d):
                all_frozen = all(
                    not p.requires_grad for p in module.parameters(recurse=False)
                )
                if all_frozen:
                    module.eval()


# ---------------------------------------------------------------------------
# Parameter groups for optimizer
# ---------------------------------------------------------------------------

def _head_param_ids(model: nn.Module, cfg: ModelConfig) -> set[int]:
    return {id(p) for p in model.head.parameters()}


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
