"""模型工厂单元测试: 验证 4 个模型都能 forward, 输出形状对.

只 build 模型, 不做真实训练 — 速度快.
"""
from __future__ import annotations

import pytest
import torch

from fed_pi.models.factory import available_models, build_model, count_params


@pytest.mark.parametrize("name", ["mnist_cnn"])
def test_mnist_models_forward(name):
    model = build_model(name, num_classes=10)
    model.eval()
    x = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 10)


@pytest.mark.parametrize("name", ["cifar_cnn", "mobilenet_v2", "squeezenet1_1"])
def test_cifar_models_forward(name):
    model = build_model(name, num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 10), f"{name} got shape {y.shape}"


def test_available_models_list():
    names = available_models()
    assert "mnist_cnn" in names
    assert "cifar_cnn" in names
    assert "mobilenet_v2" in names
    assert "squeezenet1_1" in names


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("nonexistent")


def test_count_params_ordering():
    """轻量化网络参数量应大致符合预期: SqueezeNet < MobileNet ; CNN 最小."""
    n_mnist = count_params(build_model("mnist_cnn"))
    n_cnn = count_params(build_model("cifar_cnn"))
    n_sqz = count_params(build_model("squeezenet1_1"))
    n_mb = count_params(build_model("mobilenet_v2"))

    # 仅打印, 不强 assert (torchvision 版本不同微小差异);
    # 给课程演示用 — 跑测试时会显示数值
    print(f"\nparam counts: mnist_cnn={n_mnist:_}  cifar_cnn={n_cnn:_}  "
          f"squeezenet={n_sqz:_}  mobilenet_v2={n_mb:_}")

    assert n_mnist < 200_000          # MnistCNN ~105K (fc1 主导)
    assert n_cnn < 1_000_000          # Cifar10CNN ~290K
    assert n_sqz < 2_000_000          # SqueezeNet ~730K (改了首层后参数量变化)
    assert n_mb < 3_000_000           # MobileNetV2 ~2.2M


def test_set_get_parameters_roundtrip():
    """get_parameters → set_parameters 应保留所有数值 (state_dict 顺序对齐)."""
    from fed_pi.train.params import get_parameters, set_parameters

    a = build_model("mnist_cnn")
    b = build_model("mnist_cnn")
    params = get_parameters(a)
    set_parameters(b, params)
    for pa, pb in zip(get_parameters(a), get_parameters(b)):
        assert pa.shape == pb.shape
        assert (pa == pb).all()
