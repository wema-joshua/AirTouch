"""Tests for gestures.state_machine — synthetic GestureResult sequences,
no camera or feature extraction needed."""

from __future__ import annotations

from gestures.gesture_types import GestureName, GestureResult, LifecycleState
from gestures.state_machine import GestureStateMachine

HOLD_FRAMES = 3
PINCH_HOLD_TIME = 0.25


def make_machine() -> GestureStateMachine:
    return GestureStateMachine(
        min_confidence=0.6, hold_frames=HOLD_FRAMES, pinch_hold_time=PINCH_HOLD_TIME
    )


def result(gesture: GestureName, t: float, confidence: float = 1.0) -> GestureResult:
    return GestureResult(gesture=gesture, confidence=confidence, timestamp=t)


def test_gesture_does_not_start_until_hold_frames_reached():
    machine = make_machine()
    events = []
    for i in range(HOLD_FRAMES - 1):
        events.extend(machine.update(result(GestureName.INDEX_POINT, t=i * 0.03)))
    assert events == []


def test_gesture_starts_after_hold_frames():
    machine = make_machine()
    events = []
    for i in range(HOLD_FRAMES):
        events.extend(machine.update(result(GestureName.INDEX_POINT, t=i * 0.03)))
    started = [e for e in events if e.state == LifecycleState.STARTED]
    assert len(started) == 1
    assert started[0].gesture == GestureName.INDEX_POINT


def test_gesture_continues_as_active_after_starting():
    machine = make_machine()
    for i in range(HOLD_FRAMES):
        machine.update(result(GestureName.INDEX_POINT, t=i * 0.03))
    events = machine.update(result(GestureName.INDEX_POINT, t=1.0))
    assert len(events) == 1
    assert events[0].state == LifecycleState.ACTIVE
    assert events[0].gesture == GestureName.INDEX_POINT


def test_quick_pinch_release_before_hold_time_yields_pinch_released_not_drag():
    machine = make_machine()
    t = 0.0
    events = []
    # Confirm PINCH as stable.
    for i in range(HOLD_FRAMES):
        t = i * 0.02
        events.extend(machine.update(result(GestureName.PINCH, t=t)))
    # Release quickly (well under pinch_hold_time) by sending NONE frames.
    for i in range(HOLD_FRAMES):
        t += 0.02
        events.extend(machine.update(result(GestureName.NONE, t=t, confidence=0.0)))

    releases = [e for e in events if e.state == LifecycleState.RELEASED]
    assert len(releases) == 1
    assert releases[0].gesture == GestureName.PINCH
    # It must never have escalated to a drag.
    assert all(e.gesture != GestureName.PINCH_HOLD for e in events)


def test_pinch_held_past_hold_time_escalates_to_pinch_hold():
    machine = make_machine()
    t = 0.0
    events = []
    for i in range(HOLD_FRAMES):
        t = i * 0.02
        events.extend(machine.update(result(GestureName.PINCH, t=t)))
    # Keep sending PINCH frames until past the hold-time threshold.
    while t < PINCH_HOLD_TIME + 0.05:
        t += 0.02
        events.extend(machine.update(result(GestureName.PINCH, t=t)))

    hold_started = [
        e for e in events if e.gesture == GestureName.PINCH_HOLD and e.state == LifecycleState.STARTED
    ]
    assert len(hold_started) == 1
    # PINCH itself should never have been RELEASED (it became a drag, not a click).
    pinch_released = [
        e for e in events if e.gesture == GestureName.PINCH and e.state == LifecycleState.RELEASED
    ]
    assert pinch_released == []


def test_pinch_hold_release_emits_pinch_hold_released():
    machine = make_machine()
    t = 0.0
    for i in range(HOLD_FRAMES):
        t = i * 0.02
        machine.update(result(GestureName.PINCH, t=t))
    while t < PINCH_HOLD_TIME + 0.05:
        t += 0.02
        machine.update(result(GestureName.PINCH, t=t))

    events = []
    for i in range(HOLD_FRAMES):
        t += 0.02
        events.extend(machine.update(result(GestureName.NONE, t=t, confidence=0.0)))

    released = [
        e for e in events if e.gesture == GestureName.PINCH_HOLD and e.state == LifecycleState.RELEASED
    ]
    assert len(released) == 1


def test_swipe_passes_through_immediately_without_debounce():
    machine = make_machine()
    events = machine.update(result(GestureName.SWIPE_LEFT, t=0.0))
    assert len(events) == 1
    assert events[0].gesture == GestureName.SWIPE_LEFT
    assert events[0].state == LifecycleState.ACTIVE


def test_brief_noise_does_not_cancel_stable_gesture():
    machine = make_machine()
    for i in range(HOLD_FRAMES):
        machine.update(result(GestureName.INDEX_POINT, t=i * 0.03))
    # A single noisy frame of a different gesture, below hold_frames -
    # should not cancel INDEX_POINT.
    events = machine.update(result(GestureName.FIST, t=1.0, confidence=0.9))
    assert len(events) == 1
    assert events[0].gesture == GestureName.INDEX_POINT
    assert events[0].state == LifecycleState.ACTIVE


def test_reset_clears_state():
    machine = make_machine()
    for i in range(HOLD_FRAMES):
        machine.update(result(GestureName.INDEX_POINT, t=i * 0.03))
    machine.reset()
    events = machine.update(result(GestureName.INDEX_POINT, t=100.0))
    # First frame after reset should not immediately start a gesture.
    assert events == []
