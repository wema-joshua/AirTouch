"""Interaction-mode tracking and normalized-to-screen mapping helpers.

Kept separate from ``app.application`` so the mapping math (which has
real logic worth testing: deadzones, smoothing, sensitivity) isn't
buried inside the OpenCV main loop.
"""

from __future__ import annotations

from gestures.gesture_types import GestureEvent, GestureName, InteractionMode, LifecycleState
from utils.geometry import clamp, distance
from utils.smoothing import ExponentialSmoother, Point2DSmoother


class InteractionModeController:
    """Tracks ACTIVE / PAUSED / STOPPING based on the OPEN_HAND gesture
    and explicit stop requests. Kept intentionally simple: OPEN_HAND
    pauses for as long as it's held, anything else resumes."""

    def __init__(self) -> None:
        self._mode = InteractionMode.ACTIVE

    @property
    def mode(self) -> InteractionMode:
        return self._mode

    def request_stop(self) -> None:
        self._mode = InteractionMode.STOPPING

    def on_gesture_event(self, event: GestureEvent) -> None:
        if self._mode == InteractionMode.STOPPING:
            return
        if event.gesture == GestureName.OPEN_HAND and event.state in (
            LifecycleState.STARTED,
            LifecycleState.ACTIVE,
        ):
            self._mode = InteractionMode.PAUSED
        elif event.gesture == GestureName.OPEN_HAND and event.state in (
            LifecycleState.RELEASED,
            LifecycleState.CANCELLED,
        ):
            self._mode = InteractionMode.ACTIVE

    def on_tracking_lost(self) -> None:
        if self._mode != InteractionMode.STOPPING:
            self._mode = InteractionMode.ACTIVE


class CursorMapper:
    """Maps a normalized (0..1) hand-space point to smoothed screen pixels.

    - ``sensitivity`` scales movement range around the center of the
      frame: >1.0 means a smaller physical hand movement covers the
      full screen; <1.0 requires larger movements.
    - ``deadzone`` (normalized units) suppresses tiny jitter: if the
      raw point hasn't moved further than this since the last sample,
      the previous mapped position is reused.
    - Exponential smoothing is applied on top for a natural glide.
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        sensitivity: float,
        deadzone: float,
        smoothing: float,
    ) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sensitivity = sensitivity
        self.deadzone = deadzone
        self._smoother = Point2DSmoother(factor=smoothing)
        self._last_raw: tuple[float, float] | None = None
        self._last_output: tuple[float, float] | None = None

    def reset(self) -> None:
        self._smoother.reset()
        self._last_raw = None
        self._last_output = None

    def update(self, norm_point: tuple[float, float]) -> tuple[float, float]:
        if self._last_raw is not None and distance(norm_point, self._last_raw) < self.deadzone:
            point_to_use = self._last_raw
        else:
            point_to_use = norm_point
            self._last_raw = norm_point

        scaled_x = clamp(0.5 + (point_to_use[0] - 0.5) * self.sensitivity, 0.0, 1.0)
        scaled_y = clamp(0.5 + (point_to_use[1] - 0.5) * self.sensitivity, 0.0, 1.0)

        smoothed = self._smoother.update((scaled_x, scaled_y))
        screen_x = smoothed[0] * self.screen_width
        screen_y = smoothed[1] * self.screen_height
        self._last_output = (screen_x, screen_y)
        return self._last_output


class ScrollMapper:
    """Converts vertical hand velocity into discrete scroll tick amounts."""

    def __init__(self, sensitivity: float, deadzone: float, invert: bool) -> None:
        self.sensitivity = sensitivity
        self.deadzone = deadzone
        self.invert = invert
        self._smoother = ExponentialSmoother(factor=0.4)

    def reset(self) -> None:
        self._smoother.reset()

    def update(self, velocity_y: float) -> int:
        smoothed_vy = self._smoother.update(velocity_y)
        if abs(smoothed_vy) < self.deadzone:
            return 0
        direction = 1 if smoothed_vy < 0 else -1  # moving up (negative y) -> scroll up (+1)
        if self.invert:
            direction *= -1
        magnitude = int(min(abs(smoothed_vy) * self.sensitivity, 20))
        return direction * max(magnitude, 1)
