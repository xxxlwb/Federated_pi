"""系统资源监控: 测一段代码块的 CPU 平均占用、内存峰值、墙钟时间.

设计为上下文管理器, 在 `fit` / `evaluate` 里包裹本地训练即可.
开一个后台线程做采样, 退出时计算 peak/avg.

注意:
- 不是高精度 profiler, 目标只是给课程实验提供"够用"的能耗代理指标
- 树莓派 CPU 通常 4 核, cpu_percent 返回 0~400%, 这里转成 0~100% 单核等价
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class _Sample:
    cpu_pct: list[float] = field(default_factory=list)   # 0~100 (单核等价)
    rss_mb: list[float] = field(default_factory=list)    # MB


class SysMonitor:
    """采样 CPU/内存的上下文管理器.

    Example:
        with SysMonitor(interval=0.5) as mon:
            train(...)
        print(mon.report)   # {"train_time": ..., "cpu_avg": ..., "mem_peak_mb": ...}
    """

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._proc = psutil.Process()
        self._n_cores = max(1, psutil.cpu_count(logical=True) or 1)
        self._samples = _Sample()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._t1 = 0.0

    # ---------- context manager ----------
    def __enter__(self) -> "SysMonitor":
        # 先调用一次 cpu_percent(None) 以建立基线, 否则第一次读会是 0.0
        self._proc.cpu_percent(None)
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2)
        self._t1 = time.perf_counter()

    # ---------- internal ----------
    def _loop(self) -> None:
        while not self._stop.is_set():
            # cpu_percent 是该进程相对所有核的 %, 除以核数得到单核等价
            cpu = self._proc.cpu_percent(None) / self._n_cores
            mem = self._proc.memory_info().rss / (1024 * 1024)
            self._samples.cpu_pct.append(cpu)
            self._samples.rss_mb.append(mem)
            self._stop.wait(self.interval)

    # ---------- report ----------
    @property
    def report(self) -> dict[str, float]:
        cpu_avg = (
            sum(self._samples.cpu_pct) / len(self._samples.cpu_pct)
            if self._samples.cpu_pct
            else 0.0
        )
        mem_peak = max(self._samples.rss_mb) if self._samples.rss_mb else 0.0
        return {
            "wall_time": round(self._t1 - self._t0, 4),
            "cpu_avg": round(cpu_avg, 2),
            "mem_peak_mb": round(mem_peak, 2),
        }
