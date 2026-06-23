"""模型工厂: 按名字返回模型实例.

新增模型只需在这里 register 一行即可, 不必改 client.py / server.py.
"""
from __future__ import annotations

from typing import Callable

import torch.nn as nn

from fed_pi.models.cnn import Cifar10CNN, MnistCNN
from fed_pi.models.mobilenet import build_mobilenet_v2
from fed_pi.models.squeezenet import build_squeezenet1_1

# name -> 构造函数 (signature: num_classes -> nn.Module)
_REGISTRY: dict[str, Callable[[int], nn.Module]] = {
    "mnist_cnn": MnistCNN,
    "cifar_cnn": Cifar10CNN,
    "mobilenet_v2": build_mobilenet_v2,
    "squeezenet1_1": build_squeezenet1_1,
}


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """按名字构造模型.

    Args:
        name: mnist_cnn | cifar_cnn | mobilenet_v2 | squeezenet1_1
        num_classes: 分类数 (MNIST/CIFAR-10 都是 10)
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown model {name!r}, choose from {list(_REGISTRY)}")
    return _REGISTRY[name](num_classes)


def available_models() -> list[str]:
    """列出所有支持的模型名."""
    return list(_REGISTRY)


def count_params(model: nn.Module) -> int:
    """统计可训练参数数量, 用于轻量化对比报告."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
