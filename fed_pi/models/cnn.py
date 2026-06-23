"""教学用 CNN 模型.

- MnistCNN: 2 卷积块 + 2 FC, ~22K 参数; MNIST 单通道 28x28
- Cifar10CNN: 3 个 Conv-BN-ReLU-Pool 块 + GAP + FC, ~100K 参数; CIFAR-10 三通道 32x32
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class MnistCNN(nn.Module):
    """教学版 MNIST CNN.

    结构 (输入 [B, 1, 28, 28]):
        Conv(1→16, 3x3) → ReLU → MaxPool(2)   →  [B, 16, 14, 14]
        Conv(16→32, 3x3) → ReLU → MaxPool(2)  →  [B, 32, 7, 7]
        Flatten → FC(32*7*7 → 64) → ReLU → FC(64 → num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Cifar10CNN(nn.Module):
    """适配 CIFAR-10 的中等深度 CNN, 作为发挥部分的基线.

    结构 (输入 [B, 3, 32, 32]):
        Block1: Conv(3→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool   → [B, 32, 16, 16]
        Block2: Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool  → [B, 64, 8, 8]
        Block3: Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool → [B, 128, 4, 4]
        GAP → FC(128 → num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = self._make_block(3, 32)
        self.block2 = self._make_block(32, 64)
        self.block3 = self._make_block(64, 128)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    @staticmethod
    def _make_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)
