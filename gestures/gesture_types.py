"""Shared types for the gesture engine.

Kept in their own module (rather than inside recognizer.py or
state_machine.py) so static_gestures, motion_gestures, recognizer, and
state_machine can all depend on them without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GestureName(str, Enum):
    NONE = "NONE"
    OPEN_HAND = "OPEN_HAND"
    FIST = "FIST"
    INDEX_POINT = "INDEX_POINT"
    PINCH = "PINCH"
    PINCH_HOLD = "PINCH_HOLD"
    TWO_FINGER = "TWO_FINGER"
    SWIPE_LEFT = "SWIPE_LEFT"
    SWIPE_RIGHT = "SWIPE_RIGHT"
    SWIPE_UP = "SWIPE_UP"
    SWIPE_DOWN = "SWIPE_DOWN"


class LifecycleState(str, Enum):
    """Lifecycle of a gesture as tracked by the state machine."""

    STARTED = "STARTED"
    ACTIVE = "ACTIVE"
    RELEASED = "RELEASED"
    CANCELLED = "CANCELLED"


class InteractionMode(str, Enum):
    """Top-level interaction mode, independent of any single gesture."""

    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"


@dataclass(frozen=True)
class GestureResult:
    """Output of the recognizer for a single frame: "what does this frame
    look like", with no notion of start/active/end yet."""

    gesture: GestureName
    confidence: float
    palm_center: tuple[float, float] = (0.0, 0.0)
    index_tip: tuple[float, float] = (0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)
    thumb_index_distance: float = 0.0
    timestamp: float = 0.0


@dataclass(frozen=True)
class GestureEvent:
    """Output of the state machine: a gesture plus its lifecycle stage,
    ready to be mapped to an action."""

    gesture: GestureName
    state: LifecycleState
    confidence: float
    palm_center: tuple[float, float] = (0.0, 0.0)
    index_tip: tuple[float, float] = (0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)
    timestamp: float = 0.0
