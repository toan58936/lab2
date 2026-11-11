"""Smoke test tối thiểu cho gói going_modular.

Không cần mạng, không cần GPU, không tải CIFAR-10 hay ImageNet weights.
Chỉ kiểm tra import môi trường và một hành vi đơn giản, ổn định của package.
"""

import torch

import going_modular
from going_modular import data_setup, engine, evaluate, model_builder, utils


def test_package_imports():
    """Kiểm tra package và các submodule được import thành công."""
    assert going_modular is not None
    assert data_setup is not None
    assert engine is not None
    assert evaluate is not None
    assert model_builder is not None
    assert utils is not None


def test_flat_imports_available():
    """Kiểm tra một số flat export hiện có của package."""
    assert callable(going_modular.build_model)
    assert callable(going_modular.train)
    assert callable(going_modular.train_step)
    assert callable(going_modular.evaluate_step)
    assert callable(going_modular.create_train_val_test_dataloaders)
    assert callable(going_modular.set_seed)
    assert callable(going_modular.get_device)


def test_contract_13_exports():
    """Kiểm tra 13 symbols xuất phẳng theo hợp đồng 00_lab2_description.md mục 2.8.6."""
    contract_13 = [
        "ModelConfig",
        "build_model",
        "count_params",
        "MyModel",
        "set_train_mode",
        "get_param_groups",
        "create_train_val_test_dataloaders",
        "train",
        "evaluate_step",
        "set_seed",
        "get_device",
        "run_inference",
        "evaluate_predictions",
    ]
    for name in contract_13:
        assert name in going_modular.__all__, f'{name} không có trong __all__'
        assert hasattr(going_modular, name), f'{name} không thể import từ going_modular'


def test_set_seed_reproducible():
    """Kiểm tra set_seed tạo lại cùng dãy số trên cùng môi trường."""
    going_modular.set_seed(42)
    first = torch.randn(5)
    going_modular.set_seed(42)
    second = torch.randn(5)
    assert torch.equal(first, second)
