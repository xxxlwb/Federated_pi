"""Flower 客户端入口.

每个 client 是一个独立进程:
    python -m fed_pi.client --config configs/xxx.yaml --cid 0

流程:
    1. 读 cfg
    2. 取 dataset, 按 cid 算出本地数据索引
    3. 用工厂建模型
    4. 包成 FlowerClient, 连接 server

注意:
- start_numpy_client 自 1.7 起 deprecated, 用 to_client() + start_client 替代
- batch_size, lr 等所有超参都从 cfg 读, 不在代码里硬编码
"""
from __future__ import annotations

import argparse

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fed_pi.data.datasets import get_dataset
from fed_pi.data.partition import get_client_indices
from fed_pi.models.factory import build_model
from fed_pi.train.params import get_parameters, set_parameters
from fed_pi.train.trainer import evaluate, local_train
from fed_pi.utils.config import load_config
from fed_pi.utils.logger import get_logger
from fed_pi.utils.sysmon import SysMonitor


class FlowerClient(fl.client.NumPyClient):
    """单个联邦学习客户端.

    server 调用顺序 (每一轮):
        get_parameters → fit → evaluate
    """

    def __init__(self, cid: int, model, trainloader, testloader, cfg: dict):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        self.model.to(self.device)
        self.log = get_logger(f"client-{cid}")
        self.log.info(
            f"ready, n_train={len(trainloader.dataset)}, n_test={len(testloader.dataset)}"
        )

    # ----- Flower API -----
    def get_parameters(self, config):  # noqa: ARG002 (Flower 要求该签名)
        return get_parameters(self.model)

    def fit(self, parameters, config):  # noqa: ARG002
        """server 把全局参数下发 → 本地训练 → 上传新参数 + 监控指标."""
        set_parameters(self.model, parameters)
        with SysMonitor(interval=0.5) as mon:
            wall, train_loss = local_train(
                self.model,
                self.trainloader,
                epochs=self.cfg["local_epochs"],
                lr=self.cfg["learning_rate"],
                device=self.device,
            )
        self.log.info(
            f"fit done: loss={train_loss:.4f} time={wall:.1f}s "
            f"cpu={mon.report['cpu_avg']:.1f}% mem={mon.report['mem_peak_mb']:.1f}MB"
        )
        # 返回: (新参数, 样本数, 额外指标) — 指标会被 server 端 weighted_avg_metrics 聚合
        return (
            get_parameters(self.model),
            len(self.trainloader.dataset),
            {
                "train_loss": float(train_loss),
                "train_time": float(wall),
                "cpu_avg": float(mon.report["cpu_avg"]),
                "mem_peak_mb": float(mon.report["mem_peak_mb"]),
                "cid": int(self.cid),
            },
        )

    def evaluate(self, parameters, config):  # noqa: ARG002
        """评估当前全局模型在本地测试集上的表现."""
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.testloader, self.device)
        self.log.info(f"evaluate: loss={loss:.4f} acc={acc:.4f}")
        return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML 配置路径")
    ap.add_argument("--cid", type=int, required=True, help="客户端 id, 0 ~ num_clients-1")
    ap.add_argument(
        "--server_address",
        default=None,
        help="覆盖 cfg 里的 server_address, 真机部署时用 (例: 192.168.1.100:8080)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    # 固定随机数, 保证多客户端在调用 partition 时得到一致划分
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    server_addr = args.server_address or cfg["server_address"]

    # 数据
    trainset, testset = get_dataset(cfg["dataset"])
    indices = get_client_indices(trainset, cfg, args.cid)
    if not indices:
        raise RuntimeError(
            f"client {args.cid} got 0 samples; check partition config "
            f"({cfg['partition']['strategy']})"
        )
    trainloader = DataLoader(
        Subset(trainset, indices),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )
    # 测试集全部用 — 每个 client 在同样的测试集上评估当前全局模型,
    # 加权平均后即为全局准确率
    testloader = DataLoader(testset, batch_size=256, shuffle=False)

    # 模型
    model = build_model(cfg["model"], cfg["num_classes"])

    client = FlowerClient(args.cid, model, trainloader, testloader, cfg)
    fl.client.start_client(server_address=server_addr, client=client.to_client())


if __name__ == "__main__":
    main()
