"""本地训练 / 评估.

local_train: 对一个 client 跑 n epochs 的 SGD, 返回 (耗时, 最后一轮平均 loss)
evaluate: 在测试集上算 loss 与 accuracy

设计简洁: 不引入 lr scheduler / amp / clip; 课程作业够用.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def local_train(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    momentum: float = 0.9,
) -> tuple[float, float]:
    """对模型做 `epochs` 轮 SGD 训练.

    Returns:
        (耗时秒, 最后一个 epoch 的平均 loss)
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    t0 = time.perf_counter()
    last_avg_loss = 0.0

    for _ in range(epochs):
        total_loss, n_batches = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        last_avg_loss = total_loss / max(1, n_batches)

    elapsed = time.perf_counter() - t0
    return elapsed, last_avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """评估模型.

    Returns:
        (平均交叉熵 loss, 准确率 0~1)
    """
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / max(1, total), total_correct / max(1, total)
