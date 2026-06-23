"""数据集工厂: 返回 (trainset, testset).

只支持 MNIST 和 CIFAR-10, 两者通过 torchvision 自动下载到项目根的 data/ 目录.
"""
from __future__ import annotations

from pathlib import Path

from torchvision import datasets, transforms

# 项目根 / data/
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def get_dataset(name: str, data_root: Path | str | None = None):
    """构造训练集和测试集.

    Args:
        name: "mnist" 或 "cifar10"
        data_root: 数据存放目录, 默认项目根的 data/

    Returns:
        (trainset, testset)
    """
    root = Path(data_root) if data_root else DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)
    name = name.lower()

    if name == "mnist":
        # MNIST 单通道 28x28, 标准化用官方均值/方差
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = datasets.MNIST(root=str(root), train=True, download=True, transform=tfm)
        testset = datasets.MNIST(root=str(root), train=False, download=True, transform=tfm)
        return trainset, testset

    if name == "cifar10":
        # CIFAR-10 三通道 32x32, 训练集加简单数据增强 (随机翻转/裁剪) 提高效果
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_tfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        trainset = datasets.CIFAR10(root=str(root), train=True, download=True, transform=train_tfm)
        testset = datasets.CIFAR10(root=str(root), train=False, download=True, transform=test_tfm)
        return trainset, testset

    raise ValueError(f"unsupported dataset: {name!r} (use mnist | cifar10)")


def get_targets(dataset) -> list[int]:
    """统一接口取出标签 list, 兼容 torchvision MNIST(.targets is Tensor) 与 CIFAR10(.targets is list)."""
    t = dataset.targets
    if hasattr(t, "tolist"):
        return t.tolist()
    return list(t)
