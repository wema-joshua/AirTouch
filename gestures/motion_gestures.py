"""Motion gesture detection: swipes derived from a short temporal buffer
of palm-center positions.

Designed so all four swipe directions are supported even though V1 only
wires LEFT/RIGHT to actions — adding UP/DOWN actions later needs no
changes here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from gestures.gesture_types import GestureName


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
