"""Motion gesture detection: swipes derived from a short temporal buffer
of palm-center positions.

Designed so all four swipe directions are supported even though V1 only
wires LEFT/RIGHT to actions — adding UP/DOWN actions later needs no
changes here.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from gestures.gesture_types import GestureName
from utils.smoothing import OneEuroFilter


@dataclass(frozen=True)
class _Sample:
    timestamp: float
    position: tuple[float, float]


class SwipeDetector:
    """Stateful detector: feed it (timestamp, position) samples via
    ``update`` and it returns a swipe GestureName when one completes.

    A swipe is detected by looking at the displacement between the
    oldest sample still within ``max_duration`` and the newest sample,
    provided that displacement exceeds ``min_distance`` in a dominant
    axis and the implied average velocity exceeds ``min_velocity``.
    A cooldown prevents one physical swipe from firing multiple events.
    """

    def __init__(
        self,
        min_distance: float,
        max_duration: float,
        min_velocity: float,
        cooldown: float,
    ) -> None:
        self.min_distance = min_distance
        self.max_duration = max_duration
        self.min_velocity = min_velocity
        self.cooldown = cooldown
        self._buffer: deque[_Sample] = deque()
        self._last_trigger_time: float = -float("inf")

    def reset(self) -> None:
        self._buffer.clear()
        self._last_trigger_time = -float("inf")

    def update(
        self, timestamp: float, position: tuple[float, float]
    ) -> GestureName | None:
        self._buffer.append(_Sample(timestamp, position))
        self._trim(timestamp)

        if timestamp - self._last_trigger_time < self.cooldown:
            return None
        if len(self._buffer) < 2:
            return None

        oldest = self._buffer[0]
        newest = self._buffer[-1]
        duration = newest.timestamp - oldest.timestamp
        if duration <= 0:
            return None

        dx = newest.position[0] - oldest.position[0]
        dy = newest.position[1] - oldest.position[1]

        if abs(dx) >= abs(dy):
            distance_val, velocity = abs(dx), abs(dx) / duration
            direction_positive = dx > 0
            axis = "x"
        else:
            distance_val, velocity = abs(dy), abs(dy) / duration
            direction_positive = dy > 0
            axis = "y"

        if distance_val < self.min_distance or velocity < self.min_velocity:
            return None

        self._last_trigger_time = timestamp
        self._buffer.clear()

        if axis == "x":
            return GestureName.SWIPE_RIGHT if direction_positive else GestureName.SWIPE_LEFT
        return GestureName.SWIPE_DOWN if direction_positive else GestureName.SWIPE_UP

    def _trim(self, now: float) -> None:
        while self._buffer and (now - self._buffer[0].timestamp) > self.max_duration:
            self._buffer.popleft()


# --- Continuous-motion deltas (orbit / pan / zoom) ------------------------


class DeltaTracker:
    """Turns a stream of 2D positions into smoothed frame-to-frame deltas.

    Use one tracker per controlled quantity (e.g. one for the orbit hand,
    one for the two-hand midpoint). Call ``reset`` whenever the clutch
    releases or the hand is lost, so the first frame after re-engaging
    returns (0, 0) instead of a jump from the old position.

    ``deadzone`` is in normalized image units. Movement smaller than this
    is withheld, not discarded: the reference point stays put until the
    accumulated displacement exceeds the deadzone, so slow deliberate
    motion still gets through.
    """

    def __init__(
        self,
        smooth: bool = True,
        min_cutoff: float = 1.5,
        beta: float = 8.0,
        deadzone: float = 0.0,
    ) -> None:
        self._filter_x = OneEuroFilter(min_cutoff, beta) if smooth else None
        self._filter_y = OneEuroFilter(min_cutoff, beta) if smooth else None
        self.deadzone = deadzone
        self._ref: tuple[float, float] | None = None

    def reset(self) -> None:
        if self._filter_x is not None:
            self._filter_x.reset()
        if self._filter_y is not None:
            self._filter_y.reset()
        self._ref = None

    def update(
        self, timestamp: float, position: tuple[float, float]
    ) -> tuple[float, float]:
        x, y = position
        if self._filter_x is not None and self._filter_y is not None:
            x = self._filter_x(timestamp, x)
            y = self._filter_y(timestamp, y)

        if self._ref is None:
            self._ref = (x, y)
            return (0.0, 0.0)

        dx = x - self._ref[0]
        dy = y - self._ref[1]
        if self.deadzone > 0.0 and math.hypot(dx, dy) < self.deadzone:
            return (0.0, 0.0)

        self._ref = (x, y)
        return (dx, dy)


class ScalarDeltaTracker:
    """Same as ``DeltaTracker`` for a single value, e.g. the normalized
    distance between two hands (two-hand zoom)."""

    def __init__(
        self,
        smooth: bool = True,
        min_cutoff: float = 1.5,
        beta: float = 8.0,
        deadzone: float = 0.0,
    ) -> None:
        self._filter = OneEuroFilter(min_cutoff, beta) if smooth else None
        self.deadzone = deadzone
        self._ref: float | None = None

    def reset(self) -> None:
        if self._filter is not None:
            self._filter.reset()
        self._ref = None

    def update(self, timestamp: float, value: float) -> float:
        if self._filter is not None:
            value = self._filter(timestamp, value)

        if self._ref is None:
            self._ref = value
            return 0.0

        delta = value - self._ref
        if self.deadzone > 0.0 and abs(delta) < self.deadzone:
            return 0.0

        self._ref = value
        return delta