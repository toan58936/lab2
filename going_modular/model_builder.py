"""Pretrained image classifiers adapted to the CIFAR-10 dataset."""

import torch
from torch import nn
from torchvision import models
from torchvision.models import VGG16_Weights, ResNet18_Weights


class CIFAR10ModelV0(nn.Module):

    def __init__(self, num_classes: int = 10, freeze_backbone: bool = True) -> None:
        super().__init__()
        # Tải VGG16 với trọng số pretrained ImageNet
        self.model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# V1 — ResNet-18 pretrained: Transfer Learning
# ---------------------------------------------------------------------------

class CIFAR10ModelV1(nn.Module):

    def __init__(self, num_classes: int = 10, freeze_backbone: bool = True) -> None:
        super().__init__()
        # Tải ResNet-18 với trọng số pretrained ImageNet
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Đóng băng toàn bộ backbone (mọi layer trừ fc)
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

        # Thay lớp fc cuối cùng
        in_features = self.model.fc.in_features          # 512
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

MODEL_NAMES = ("v0", "v1")


def build_model(name: str,
                num_classes: int = 10,
                freeze_backbone: bool = True) -> nn.Module:

    name = name.lower()

    if name == "v0":
        return CIFAR10ModelV0(num_classes=num_classes, freeze_backbone=freeze_backbone)

    if name == "v1":
        return CIFAR10ModelV1(num_classes=num_classes, freeze_backbone=freeze_backbone)

    raise ValueError(
        f"Unknown model name '{name}'. Choose one of {MODEL_NAMES}."
    )


def count_trainable_params(model: nn.Module) -> int:

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:

    return sum(p.numel() for p in model.parameters())
