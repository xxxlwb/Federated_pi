"""跨实验对比图生成器.

用法:
    python scripts/plot_comparison.py --group noniid        # E2/E3/E4 同框对比
    python scripts/plot_comparison.py --group lightweight   # E6/E7/E8 同框对比
    python scripts/plot_comparison.py --group all           # 上面两组都画

输出到 results/_comparison/ 目录.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 让脚本可以从项目根直接 python scripts/xxx 启动
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fed_pi.utils.metrics import load_metrics, plot_comparison  # noqa: E402

OUT_DIR = ROOT / "results" / "_comparison"

GROUPS = {
    "noniid": [
        "mnist_noniid_a01",
        "mnist_noniid_a05",
        "mnist_noniid_a10",
    ],
    "lightweight": [
        "cifar10_cnn",
        "cifar10_mobilenet",
        "cifar10_squeezenet",
    ],
}


def _filter_existing(names: list[str]) -> list[str]:
    """只保留已经跑过的实验 (results/<name>/metrics.json 存在)."""
    keep = []
    for n in names:
        if (ROOT / "results" / n / "metrics.json").exists():
            keep.append(n)
        else:
            print(f"[skip] {n}: metrics.json not found (实验未跑)", file=sys.stderr)
    return keep


def plot_noniid() -> None:
    exps = _filter_existing(GROUPS["noniid"])
    if not exps:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_comparison(exps, OUT_DIR / "noniid_acc.png",
                    title="Non-IID (Dirichlet α) — accuracy", metric="accuracy")
    plot_comparison(exps, OUT_DIR / "noniid_loss.png",
                    title="Non-IID (Dirichlet α) — loss", metric="loss")
    print(f"saved {OUT_DIR / 'noniid_acc.png'} and noniid_loss.png")


def plot_lightweight() -> None:
    exps = _filter_existing(GROUPS["lightweight"])
    if not exps:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 准确率对比
    plot_comparison(exps, OUT_DIR / "lightweight_acc.png",
                    title="Model comparison — accuracy", metric="accuracy")

    # 2) 训练时间 / 内存峰值 — 取每轮 fit 阶段平均
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, ylabel in [
        (axes[0], "train_time", "fit time (s)"),
        (axes[1], "mem_peak_mb", "memory peak (MB)"),
    ]:
        for name in exps:
            m = load_metrics(name)
            pairs = m["metrics_distributed_fit"].get(key, [])
            if not pairs:
                continue
            xs = [r for r, _ in pairs]
            ys = [v for _, v in pairs]
            ax.plot(xs, ys, "o-", label=name)
        ax.set_xlabel("round")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Lightweight network comparison — resources")
    fig.tight_layout()
    p = OUT_DIR / "lightweight_resources.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"saved {p}")

    # 3) 文字汇总 (最终准确率 + 平均训练时间 + 内存峰值)
    summary = {}
    for name in exps:
        m = load_metrics(name)
        acc = m["metrics_distributed"].get("accuracy", [])
        tt = m["metrics_distributed_fit"].get("train_time", [])
        mem = m["metrics_distributed_fit"].get("mem_peak_mb", [])
        summary[name] = {
            "final_acc": acc[-1][1] if acc else None,
            "avg_fit_time": sum(v for _, v in tt) / len(tt) if tt else None,
            "peak_mem_mb": max(v for _, v in mem) if mem else None,
        }
    out_json = OUT_DIR / "lightweight_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"summary saved to {out_json}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["noniid", "lightweight", "all"])
    args = ap.parse_args()

    if args.group in {"noniid", "all"}:
        plot_noniid()
    if args.group in {"lightweight", "all"}:
        plot_lightweight()


if __name__ == "__main__":
    main()
