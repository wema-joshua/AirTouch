"""Gesture state machine.

Turns a per-frame stream of raw ``GestureResult`` classifications into a
stream of ``GestureEvent`` objects with explicit lifecycle stages
(STARTED / ACTIVE / RELEASED / CANCELLED). This is what makes temporal
gestures (pinch-hold-to-drag) and future combinations possible without
scattering ad-hoc "was it pinched last frame?" flags across the codebase.

Pinch lifecycle (see module docstring in gestures/__init__.py and the
project spec):

    OPEN -> PINCH (STARTED) -> [hold_time elapses] -> PINCH_HOLD (STARTED)
    -> PINCH_HOLD (ACTIVE, while moving) -> released -> PINCH_HOLD (RELEASED)

    OPEN -> PINCH (STARTED) -> released before hold_time -> PINCH (RELEASED)

A quick PINCH that RELEASES before ever escalating to PINCH_HOLD is what
the action layer maps to a single LEFT_CLICK. A RELEASED PINCH_HOLD maps
to EndDrag. This way a click never fires repeatedly while held, and a
drag never fires a spurious click on release.
"""

from __future__ import annotations

from dataclasses import dataclass

from gestures.gesture_types import GestureEvent, GestureName, GestureResult, LifecycleState

_SWIPES = {
    GestureName.SWIPE_LEFT,
    GestureName.SWIPE_RIGHT,
    GestureName.SWIPE_UP,
    GestureName.SWIPE_DOWN,
}


@dataclass
class _StableGesture:
    name: GestureName
    started_at: float
    last_result: GestureResult


class GestureStateMachine:
    """Debounces raw recognizer output into a lifecycle event stream."""

    def __init__(
        self,
        min_confidence: float = 0.6,
        hold_frames: int = 3,
        pinch_hold_time: float = 0.25,
    ) -> None:
        self.min_confidence = min_confidence
        self.hold_frames = hold_frames
        self.pinch_hold_time = pinch_hold_time

        self._stable: _StableGesture | None = None
        self._pending_name: GestureName = GestureName.NONE
        self._pending_count: int = 0

    def reset(self) -> None:
        self._stable = None
        self._pending_name = GestureName.NONE
        self._pending_count = 0

    def update(self, result: GestureResult) -> list[GestureEvent]:
        # Swipes are already discrete, single-frame events from the
        # motion detector's own cooldown logic — pass them straight
        # through rather than debouncing them.
        if result.gesture in _SWIPES:
            return [self._event(result.gesture, LifecycleState.ACTIVE, result)]

        raw = result.gesture if result.confidence >= self.min_confidence else GestureName.NONE
        events: list[GestureEvent] = []

        if self._stable is not None and self._matches_stable(raw, self._stable.name):
            self._pending_name = GestureName.NONE
            self._pending_count = 0
            events.extend(self._continue_stable(result))
            return events

        # raw differs from the current stable gesture (or there is none yet).
        if raw == self._pending_name:
            self._pending_count += 1
        else:
            self._pending_name = raw
            self._pending_count = 1

        if self._pending_count >= self.hold_frames:
            if self._stable is not None:
                events.append(self._end_stable(LifecycleState.RELEASED, result))
            self._pending_name = GestureName.NONE
            self._pending_count = 0
            if raw != GestureName.NONE:
                self._stable = _StableGesture(name=raw, started_at=result.timestamp, last_result=result)
                events.append(self._event(raw, LifecycleState.STARTED, result))
            else:
                self._stable = None
        elif self._stable is not None:
            # Not yet confirmed as a real change — treat as noise and keep
            # the current stable gesture alive.
            events.extend(self._continue_stable(result))

        return events

    @staticmethod
    def _matches_stable(raw: GestureName, stable_name: GestureName) -> bool:
        """Whether a raw per-frame classification should be treated as a
        continuation of the current stable gesture. PINCH_HOLD is an
        internal escalation of PINCH that the recognizer never emits
        directly, so a continuing PINCH classification must still count
        as "still holding" once the stable gesture has escalated."""
        if raw == stable_name:
            return True
        if stable_name == GestureName.PINCH_HOLD and raw == GestureName.PINCH:
            return True
        return False

    def _continue_stable(self, result: GestureResult) -> list[GestureEvent]:
        assert self._stable is not None
        self._stable.last_result = result
        events: list[GestureEvent] = []

        if self._stable.name == GestureName.PINCH:
            elapsed = result.timestamp - self._stable.started_at
            if elapsed >= self.pinch_hold_time:
                # Escalate PINCH -> PINCH_HOLD in place. No RELEASED is
                # emitted for PINCH here: it never completed as a click,
                # it "became" a drag.
                self._stable = _StableGesture(
                    name=GestureName.PINCH_HOLD,
                    started_at=self._stable.started_at,
                    last_result=result,
                )
                events.append(self._event(GestureName.PINCH_HOLD, LifecycleState.STARTED, result))
                return events

        events.append(self._event(self._stable.name, LifecycleState.ACTIVE, result))
        return events

    def _end_stable(self, state: LifecycleState, result: GestureResult) -> GestureEvent:
        assert self._stable is not None
        ended = self._stable
        self._stable = None
        return self._event(ended.name, state, result)

    @staticmethod
    def _event(gesture: GestureName, state: LifecycleState, result: GestureResult) -> GestureEvent:
        return GestureEvent(
            gesture=gesture,
            state=state,
            confidence=result.confidence,
            palm_center=result.palm_center,
            index_tip=result.index_tip,
            velocity=result.velocity,
            timestamp=result.timestamp,
            delta=result.delta,
            pair_distance_delta=result.pair_distance_delta,
            pair_midpoint_delta=result.pair_midpoint_delta,
            pair_angle_delta_deg=result.pair_angle_delta_deg,
        )