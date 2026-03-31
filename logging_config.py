import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging(
    log_dir: str = "logs",
    log_file: str = "app.log",
    logger_name: str = "legal_app",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    统一初始化日志：
    - 控制台输出
    - 文件持久化
    - 日志轮转
    - 防止重复添加 handler
    """

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / log_file

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # 防止 Streamlit rerun 或重复 import 时重复加 handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=5 * 1024 * 1024,   # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    logger.info(
        f"[LoggerInit] logger_name={logger_name} log_file={log_path} pid={os.getpid()}"
    )
    return logger


def get_logger(name: str = "legal_app") -> logging.Logger:
    """
    获取已配置好的 logger。
    """
    return logging.getLogger(name)