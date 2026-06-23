"""MobileNetV2 适配 CIFAR-10 (32x32) 的封装.

torchvision 的 MobileNetV2 默认是给 ImageNet (224x224) 设计的:
- 首个 conv stride=2 会立刻把 32x32 缩到 16x16, 信息丢失太多
- 这里把首个 conv 的 stride 改成 1, 保留 32x32

参数量: 约 2.3M
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import mobilenet_v2


def build_mobilenet_v2(num_classes: int = 10) -> nn.Module:
    """构造适配小图的 MobileNetV2 (无预训练)."""
    model = mobilenet_v2(weights=None, num_classes=num_classes)
    # 首层 Conv stride 2 → 1, 避免 32x32 输入立即变成 16x16
    first_conv: nn.Conv2d = model.features[0][0]  # type: ignore[assignment]
    model.features[0][0] = nn.Conv2d(
        in_channels=first_conv.in_channels,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=1,
        padding=first_conv.padding,
        bias=False,
    )
    return model
