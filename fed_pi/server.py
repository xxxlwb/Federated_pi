"""Flower 服务端入口.

用法:
    python -m fed_pi.server --config configs/xxx.yaml

负责:
    1. 用 cfg.num_clients 配置 FedAvg 策略
    2. 启动 start_server 等待 client 连接
    3. 训练完成后保存 metrics.json + 画收敛曲线 PNG

FedAvg 聚合:
    全局模型参数 = Σ (n_i / N) * 本地模型参数
    其中 n_i 是 client i 的训练样本数, N = Σ n_i.
    Flower 内置 FedAvg 已实现, 我们只配置 client 数量阈值.
"""
from __future__ import annotations

import argparse

import flwr as fl

from fed_pi.utils.config import load_config
from fed_pi.utils.logger import get_logger
from fed_pi.utils.metrics import plot_curves, save_history

log = get_logger("server")


def weighted_avg_metrics(results: list[tuple[int, dict]]) -> dict[str, float]:
    """按样本数加权平均所有数值型指标.

    Args:
        results: [(num_examples_i, metrics_dict_i), ...]
            metrics_dict 里通常有 accuracy / train_loss / train_time / cpu_avg / mem_peak_mb

    Returns:
        聚合后的指标字典 (只保留 int/float 类型的 key)
    """
    if not results:
        return {}
    total = sum(n for n, _ in results)
    agg: dict[str, float] = {}
    # 取第一个 dict 的 key 作为模板 (所有 client 字段应一致)
    for k, v in results[0][1].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            agg[k] = sum(n * m.get(k, 0.0) for n, m in results) / total
    return agg


def make_strategy(cfg: dict) -> fl.server.strategy.FedAvg:
    """构造 FedAvg 策略.

    fraction_fit / fraction_evaluate = 1.0 表示每轮所有 client 都参加;
    min_*_clients = num_clients 表示必须全部就绪才开始训练,
    支持 num_clients=1 的单客户端退化场景.
    """
    n = cfg["num_clients"]
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=n,
        min_evaluate_clients=n,
        min_available_clients=n,
        fit_metrics_aggregation_fn=weighted_avg_metrics,
        evaluate_metrics_aggregation_fn=weighted_avg_metrics,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML 配置路径")
    ap.add_argument(
        "--server_address",
        default=None,
        help="覆盖 cfg 里的 server_address (例: 0.0.0.0:8082, 用于避免端口冲突)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.server_address:
        cfg["server_address"] = args.server_address
    log.info(
        f"starting exp={cfg['exp_name']} dataset={cfg['dataset']} model={cfg['model']} "
        f"clients={cfg['num_clients']} rounds={cfg['num_rounds']}"
    )
    log.info(f"listening on {cfg['server_address']}")

    strategy = make_strategy(cfg)
    history = fl.server.start_server(
        server_address=cfg["server_address"],
        config=fl.server.ServerConfig(num_rounds=cfg["num_rounds"]),
        strategy=strategy,
    )

    # 实验结束 — 保存指标 + 画曲线
    metrics_path = save_history(history, cfg["exp_name"], extra={"config": cfg})
    log.info(f"metrics saved to {metrics_path}")
    pngs = plot_curves(history, cfg["exp_name"])
    for p in pngs:
        log.info(f"curve saved to {p}")
    log.info("done.")


if __name__ == "__main__":
    main()
