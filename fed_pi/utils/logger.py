"""统一彩色日志.

每个进程 (server / client_0 / client_1 ...) 加前缀, 便于在同一终端调试时区分.
"""
from __future__ import annotations

import logging
import sys

# ANSI 颜色
_COLORS = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[1;31m",
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname:<7}{_RESET}"
        return super().format(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """获取一个带颜色的 logger.

    用法:
        log = get_logger("server")
        log.info("started")
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 已配置, 直接返回

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _ColorFormatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
