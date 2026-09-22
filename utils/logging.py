"""Structured logging setup for AirTouch.

Centralizing logger configuration means every module can do
``logger = logging.getLogger(__name__)`` and get consistent formatting,
while the level is controlled from config.yaml.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure the root 'airtouch' logger. Safe to call once at startup."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("airtouch")
    logger.setLevel(numeric_level)

    if logger.handlers:
        # Already configured (e.g. re-entrant call); avoid duplicate handlers.
        logger.setLevel(numeric_level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'airtouch' namespace."""
    if not name.startswith("airtouch"):
        name = f"airtouch.{name}"
    return logging.getLogger(name)
