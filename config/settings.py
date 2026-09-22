"""Typed configuration objects loaded from config.yaml.

Every tunable threshold in AirTouch is expected to live here rather than
being hardcoded near where it is used. This keeps behavior adjustable
without code changes and gives every layer a single, predictable source
of truth for its parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class CameraConfig:
    device: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirrored: bool = True


@dataclass(frozen=True)
class TrackingConfig:
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    max_num_hands: int = 1


@dataclass(frozen=True)
class CursorConfig:
    smoothing: float = 0.65
    sensitivity: float = 1.0
    deadzone: float = 0.02
    landmark: int = 8


@dataclass(frozen=True)
class PinchConfig:
    threshold: float = 0.05
    release_threshold: float = 0.07
    hold_time: float = 0.25


@dataclass(frozen=True)
class ScrollConfig:
    sensitivity: float = 2.0
    deadzone: float = 0.02
    invert: bool = False


@dataclass(frozen=True)
class SwipeConfig:
    min_distance: float = 0.15
    max_duration: float = 0.6
    min_velocity: float = 0.4
    cooldown: float = 0.5
    landmark: int = 0


@dataclass(frozen=True)
class GestureConfig:
    min_confidence: float = 0.6
    static_hold_frames: int = 3


@dataclass(frozen=True)
class DebugConfig:
    enabled: bool = True
    show_landmarks: bool = True
    show_fps: bool = True
    log_level: str = "INFO"


@dataclass(frozen=True)
class SafetyConfig:
    quit_keys: tuple = field(default_factory=lambda: ("q", "Q"))


@dataclass(frozen=True)
class Settings:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)
    pinch: PinchConfig = field(default_factory=PinchConfig)
    scroll: ScrollConfig = field(default_factory=ScrollConfig)
    swipe: SwipeConfig = field(default_factory=SwipeConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    @staticmethod
    def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
        value = raw.get(key, {})
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ConfigError(f"Config section '{key}' must be a mapping.")
        return value

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Settings":
        try:
            safety_raw = cls._section(raw, "safety")
            quit_keys = safety_raw.get("quit_keys", ["q", "Q"])
            return cls(
                camera=CameraConfig(**cls._section(raw, "camera")),
                tracking=TrackingConfig(**cls._section(raw, "tracking")),
                cursor=CursorConfig(**cls._section(raw, "cursor")),
                pinch=PinchConfig(**cls._section(raw, "pinch")),
                scroll=ScrollConfig(**cls._section(raw, "scroll")),
                swipe=SwipeConfig(**cls._section(raw, "swipe")),
                gesture=GestureConfig(**cls._section(raw, "gesture")),
                debug=DebugConfig(**cls._section(raw, "debug")),
                safety=SafetyConfig(quit_keys=tuple(quit_keys)),
            )
        except TypeError as exc:
            raise ConfigError(f"Invalid configuration key: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Settings":
        """Load settings from a YAML file, falling back to defaults if
        the file is missing (with a warning left to the caller's logger).
        """
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Failed to parse {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"Top level of {path} must be a mapping.")
        return cls.from_dict(raw)
