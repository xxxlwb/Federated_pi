"""SqueezeNet1_1 适配 CIFAR-10 (32x32) 的封装.

torchvision 的 SqueezeNet1_1 默认输入 224x224, 首层 conv 用 kernel=3 stride=2,
对 32x32 输入立刻变成 16x16; 这里改成 stride=1 + kernel=3 + padding=1, 保留分辨率.
同时把 classifier 的最后 Conv 改成输出 num_classes 个通道.

参数量: 约 1.2M
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import squeezenet1_1


def build_squeezenet1_1(num_classes: int = 10) -> nn.Module:
    """构造适配小图的 SqueezeNet1_1 (无预训练)."""
    model = squeezenet1_1(weights=None, num_classes=num_classes)
    # 首层 conv: kernel 3, stride 2 → stride 1, padding 1
    model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    # torchvision SqueezeNet 的 classifier 已根据 num_classes 配置, 这里无需再改
    return model
