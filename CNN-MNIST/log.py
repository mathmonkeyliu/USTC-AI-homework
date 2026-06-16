import logging
from logging import Logger
from pathlib import Path


def setup_logging(log_path: Path, logger: Logger) -> None:
    """Configure a logger to write INFO messages to a file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.propagate = False


def get_model_logger(model_name: str) -> logging.Logger:
    """Return a named logger for a specific model."""
    return logging.getLogger(f"MNIST_CNN_{model_name}")
