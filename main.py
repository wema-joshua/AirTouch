"""AirTouch entry point.

Usage:
    python main.py [--config path/to/config.yaml]
"""

from __future__ import annotations

import argparse
import sys

from app.application import Application
from config.settings import ConfigError, Settings
from utils.logging import configure_logging, get_logger


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AirTouch gesture-control platform")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    try:
        settings = Settings.load(args.config)
    except ConfigError as exc:
        # Logging isn't configured yet if the config itself failed to load.
        print(f"AirTouch failed to start: {exc}", file=sys.stderr)
        return 1

    configure_logging(settings.debug.log_level)
    logger = get_logger("main")

    try:
        app = Application(settings)
        app.run()
    except Exception as exc:  # noqa: BLE001 - top-level guard, we want to log and exit cleanly
        logger.error("AirTouch encountered a fatal error: %s", exc)
        return 1

    logger.info("AirTouch exited cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
