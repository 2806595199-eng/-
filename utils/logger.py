"""结构化日志 — JSON 格式，按天轮转，保留 30 天"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler


class JSONFormatter(logging.Formatter):
    """输出 JSON 格式日志"""

    def format(self, record):
        entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "context"):
            entry["context"] = record.context
        # 把所有非标准属性也纳入 context
        for key in ("predicted_f", "risk", "model", "pacl", "defluor",
                    "cost", "warnings", "elapsed", "n_schemes"):
            if hasattr(record, key):
                entry.setdefault("context", {})[key] = getattr(record, key)
        # 异常信息
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


class ContextLogger(logging.LoggerAdapter):
    """支持 context 字典的结构化日志"""

    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})

    def process(self, msg, kwargs):
        # 合并额外上下文，不覆盖调用者传入的 extra
        existing = kwargs.get("extra", {})
        merged = {**existing, "context": self.extra}
        kwargs["extra"] = merged
        return msg, kwargs


def setup_logger(log_dir="logs", level="INFO",
                 backup_count=30) -> logging.Logger:
    """配置根日志器 — 所有子模块自动继承

    调用后 logging.getLogger("serve") / logging.getLogger("engine") 都能输出到文件
    """
    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 避免重复添加 handler
    if not any(isinstance(h, TimedRotatingFileHandler) for h in root.handlers):
        # 控制台
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%m-%d %H:%M:%S"))
        root.addHandler(console)

        # 文件（JSON，按天轮转）
        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, "water_ai.log"),
            when="midnight", interval=1,
            backupCount=backup_count, encoding="utf-8")
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)

    return root


_initialized = False


def get_logger(module_name="defluor"):
    """获取模块日志实例，首次调用自动初始化"""
    global _initialized
    if not _initialized:
        from core import config as cfg
        setup_logger(
            log_dir=getattr(cfg, "LOG_DIR", "logs"),
            level=getattr(cfg, "LOG_LEVEL", "INFO"),
            backup_count=getattr(cfg, "LOG_RETENTION_DAYS", 30),
        )
        _initialized = True

    raw = logging.getLogger(module_name)
    return ContextLogger(raw)
