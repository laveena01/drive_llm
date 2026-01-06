"""
Centralized logging module for the LLM driving project.
Logs appear both in command line and in a single consolidated log file for tracking.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


# Global logger instance - shared across all modules
_logger: Optional[logging.Logger] = None
_log_file_path: Optional[Path] = None


def setup_logger(
    name: str = "llm_driving",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    rank: Optional[int] = None,
) -> logging.Logger:
    """
    Set up a shared logger with console and file output.
    All modules will use the same logger instance and log to the same file.

    Args:
        name: Name of the logger (root logger name)
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Path to the single consolidated log file
        rank: Optional rank for distributed training (only rank 0 logs to console by default)

    Returns:
        Configured logger instance
    """
    global _logger, _log_file_path

    # If logger already exists, return it
    if _logger is not None:
        return _logger

    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        _logger = logger
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Create formatter with module name
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (only for rank 0 in distributed setting, or always if rank is None)
    if rank is None or rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler - single consolidated log file
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        _log_file_path = log_file
        logger.info(f"Logger initialized. Log file: {log_file}")

    _logger = logger
    return logger


def get_logger(module_name: str = "llm_driving") -> logging.Logger:
    """
    Get a child logger for a specific module.
    All child loggers share the same handlers (console + file) from the root logger.

    Args:
        module_name: Name of the module (will appear in log messages)

    Returns:
        Logger instance for the module
    """
    global _logger

    # If root logger not initialized, create a basic one
    if _logger is None:
        _logger = setup_logger()

    # Return a child logger - it will inherit handlers from root
    return logging.getLogger(f"llm_driving.{module_name}")
