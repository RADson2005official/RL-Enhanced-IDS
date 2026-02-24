"""
Production-grade structured JSON logging for the RL-Enhanced IDS.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

try:
    from pythonjsonlogger.json import JsonFormatter as _JsonFormatter
except ImportError:                       # fallback for older versions
    from pythonjsonlogger.jsonlogger import JsonFormatter as _JsonFormatter


_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def setup_logger(
    name: str = "rl_ids",
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
    json_format: bool = True,
) -> logging.Logger:
    """
    Create a production-ready logger with console + optional rotating file handlers.

    Args:
        name: Logger name.
        level: Logging level string.
        log_file: Optional path for rotating file handler.
        max_bytes: Max bytes per log file before rotation.
        backup_count: Number of rotated backup files to keep.
        json_format: Whether to use structured JSON output.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # ── Formatter ───────────────────────────────────────────────────
    if json_format:
        formatter = _JsonFormatter(
            _LOG_FORMAT,
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    else:
        formatter = logging.Formatter(_LOG_FORMAT)

    # ── Console handler ─────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # ── File handler (rotating) ─────────────────────────────────────
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
