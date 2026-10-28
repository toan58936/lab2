"""DataLoader and ImageNet preprocessing helpers for CIFAR-10 transfer learning."""

from typing import Callable, List, Literal, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, VGG16_Weights


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------


def get_pretrained_transform(
    augment: bool = False,
    model_name: Literal["vgg16", "resnet18"] = "vgg16",
) -> transforms.Compose:
    """Return preprocessing associated with the selected pretrained weights.

    Args:
        augment: If True, apply a random horizontal flip before the weights transform.
        model_name: Architecture whose pretrained weights define preprocessing.

    Raises:
        ValueError: If ``model_name`` is not supported.

    Returns:
        A composed transform compatible with the selected pretrained weights.
    """
    if model_name == "vgg16":
        weights = VGG16_Weights.IMAGENET1K_V1
    elif model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Choose 'vgg16' or 'resnet18'."
        )

    base_ops = []
    if augment:
        base_ops.append(transforms.RandomHorizontalFlip())
    base_ops.append(weights.transforms())
    return transforms.Compose(base_ops)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

class _TransformSubset(Dataset):
    """Apply a split-specific transform without duplicating CIFAR-10 image data."""

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        transform: Optional[Callable],
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[object, object]:
        image, target = self.dataset[self.indices[index]]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def stratified_split_indices(
    labels: Sequence[int],
    val_size: int,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Split sample indices into train and validation sets with class balance."""
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    dataset_size = labels_tensor.numel()
    if val_size <= 0 or val_size >= dataset_size:
        raise ValueError(
            f"val_size must be between 1 and {dataset_size - 1}; got {val_size}."
        )

    class_indices = {
        int(label): torch.where(labels_tensor == label)[0]
        for label in torch.unique(labels_tensor, sorted=True)
    }
    if any(indices.numel() < 2 for indices in class_indices.values()):
        raise ValueError("Each class must have at least two samples to create a stratified split.")
    if val_size > sum(indices.numel() - 1 for indices in class_indices.values()):
        raise ValueError("val_size must leave at least one training sample in every class.")

    quotas = {
        label: val_size * indices.numel() / dataset_size
        for label, indices in class_indices.items()
    }
    val_counts = {
        label: min(int(quotas[label]), indices.numel() - 1)
        for label, indices in class_indices.items()
    }

    remaining = val_size - sum(val_counts.values())
    while remaining:
        candidates = sorted(
            (
                label
                for label, indices in class_indices.items()
                if val_counts[label] < indices.numel() - 1
            ),
            key=lambda label: (quotas[label] - int(quotas[label]), -label),
            reverse=True,
        )
        if not candidates:
            raise ValueError("Unable to allocate validation samples while keeping every class in train.")
        for label in candidates:
            if remaining == 0:
                break
            val_counts[label] += 1
            remaining -= 1

    generator = torch.Generator().manual_seed(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    for label, indices in class_indices.items():
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)].tolist()
        count = val_counts[label]
        val_indices.extend(shuffled[:count])
        train_indices.extend(shuffled[count:])

    return sorted(train_indices), sorted(val_indices)


def create_train_val_test_dataloaders(
    batch_size: int = 32,
    data_dir: str = "data",
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    val_size: int = 5000,
    seed: int = 42,
    num_workers: int = 0,
    download: bool = True,
    transform_model: Literal["vgg16", "resnet18"] = "vgg16",
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create stratified train/validation loaders and an untouched CIFAR-10 test loader.

    Train and validation use the same underlying CIFAR-10 training data but
    can have different transforms. The split is stratified and reproducible
    for a fixed ``seed``. ``transform_model`` selects the matching pretrained
    weights preset when explicit transforms are not provided.
    """
    if train_transform is None:
        train_transform = get_pretrained_transform(
            augment=False, model_name=transform_model
        )
    if eval_transform is None:
        eval_transform = get_pretrained_transform(
            augment=False, model_name=transform_model
        )

    raw_train_data = datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=None
    )
    raw_test_data = datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=None
    )
    train_indices, val_indices = stratified_split_indices(
        raw_train_data.targets,
        val_size=val_size,
        seed=seed,
    )

    train_data = _TransformSubset(raw_train_data, train_indices, train_transform)
    val_data = _TransformSubset(raw_train_data, val_indices, eval_transform)
    test_data = _TransformSubset(
        raw_test_data, range(len(raw_test_data)), eval_transform
    )
    pin_memory = torch.cuda.is_available()

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dataloader, val_dataloader, test_dataloader, raw_train_data.classes
