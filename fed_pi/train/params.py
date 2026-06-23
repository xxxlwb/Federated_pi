"""模型参数 <-> numpy 数组列表 之间的转换.

Flower 的 NumPyClient 接口要求 client 收发 list[np.ndarray] 形式的参数.
这两个函数处理 state_dict 与 list 的互转, 是 FedAvg 流程的关键纽带.

设计说明:
- 用 state_dict 而不是 parameters(), 因为前者包含 BatchNorm running_mean/var
  这些 buffer (这些参数不参与梯度更新, 但对推理至关重要)
- 顺序由 state_dict.keys() 决定, get/set 必须对应一致
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    """从模型 state_dict 提取所有参数为 numpy 数组列表."""
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """把 numpy 列表写回模型 state_dict.

    用 strict=True 校验形状一致, 若 server/client 模型不一致会立刻报错,
    避免在训练时才发现.
    """
    keys = list(model.state_dict().keys())
    if len(keys) != len(parameters):
        raise ValueError(
            f"param count mismatch: model has {len(keys)} tensors, "
            f"got {len(parameters)} arrays"
        )
    new_state = OrderedDict()
    for k, v in zip(keys, parameters):
        new_state[k] = torch.from_numpy(np.array(v))
    model.load_state_dict(new_state, strict=True)
