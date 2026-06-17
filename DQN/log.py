import logging
from logging import Logger
from pathlib import Path


def setup_logging(log_path: Path, logger: Logger) -> None:
    """Write experiment logs to one file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False


def get_logger(run_name: str) -> logging.Logger:
    """Return a named DQN logger."""
    return logging.getLogger(f"DQN_{run_name}")
