"""指标记录与可视化.

- record_round: server 每轮回调里把指标 append 进内存列表
- save_history: 把 Flower History 对象 + 自定义指标存成 metrics.json
- plot_curves: 准确率 + loss 收敛曲线 PNG
- plot_comparison: 跨实验对比 (Non-IID alpha 对比 / 模型对比)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # 无显示器环境 (树莓派 ssh / CI) 必须用 Agg
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _exp_dir(exp_name: str) -> Path:
    d = RESULTS_DIR / exp_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_history(history: Any, exp_name: str, extra: dict | None = None) -> Path:
    """把 Flower History + 额外指标存成 JSON.

    Flower History 字段说明:
    - losses_distributed:   [(round, loss), ...]  各客户端 evaluate loss 加权平均
    - metrics_distributed:  {key: [(round, val), ...]}  evaluate_metrics_aggregation_fn 返回
    - metrics_distributed_fit: {key: [(round, val), ...]}  fit_metrics_aggregation_fn 返回
    """
    data = {
        "exp_name": exp_name,
        "losses_distributed": list(history.losses_distributed),
        "metrics_distributed": {
            k: list(v) for k, v in history.metrics_distributed.items()
        },
        "metrics_distributed_fit": {
            k: list(v) for k, v in history.metrics_distributed_fit.items()
        },
    }
    if extra:
        data["extra"] = extra
    path = _exp_dir(exp_name) / "metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def plot_curves(history: Any, exp_name: str) -> list[Path]:
    """画准确率曲线与 loss 曲线."""
    outs = []
    rnds_loss = [r for r, _ in history.losses_distributed]
    vals_loss = [v for _, v in history.losses_distributed]

    acc_pairs = history.metrics_distributed.get("accuracy", [])
    rnds_acc = [r for r, _ in acc_pairs]
    vals_acc = [v for _, v in acc_pairs]

    # Loss 曲线
    if rnds_loss:
        plt.figure(figsize=(6, 4))
        plt.plot(rnds_loss, vals_loss, "o-", label="loss")
        plt.xlabel("round")
        plt.ylabel("loss")
        plt.title(f"{exp_name} — loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        p = _exp_dir(exp_name) / "curve_loss.png"
        plt.savefig(p, dpi=120)
        plt.close()
        outs.append(p)

    # Accuracy 曲线
    if rnds_acc:
        plt.figure(figsize=(6, 4))
        plt.plot(rnds_acc, vals_acc, "o-", color="tab:green", label="accuracy")
        plt.xlabel("round")
        plt.ylabel("accuracy")
        plt.title(f"{exp_name} — accuracy")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        p = _exp_dir(exp_name) / "curve_acc.png"
        plt.savefig(p, dpi=120)
        plt.close()
        outs.append(p)
    return outs


def load_metrics(exp_name: str) -> dict:
    """读取已保存的实验指标."""
    path = _exp_dir(exp_name) / "metrics.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_comparison(
    exp_names: list[str],
    out_path: Path,
    title: str = "comparison",
    metric: str = "accuracy",
) -> Path:
    """把多个实验的 metric 曲线画在一张图上.

    用于 Non-IID alpha 对比 / 轻量化模型对比.
    """
    plt.figure(figsize=(7, 5))
    for name in exp_names:
        m = load_metrics(name)
        if metric == "loss":
            pairs = m["losses_distributed"]
        else:
            pairs = m["metrics_distributed"].get(metric, [])
        if not pairs:
            continue
        xs = [r for r, _ in pairs]
        ys = [v for _, v in pairs]
        plt.plot(xs, ys, "o-", label=name)
    plt.xlabel("round")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path
