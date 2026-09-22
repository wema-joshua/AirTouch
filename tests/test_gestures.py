"""Tests for gesture classification: static poses (gestures.static_gestures)
and the combined rule-based recognizer, including swipe detection.
"""

from __future__ import annotations

from gestures.features import FeatureExtractor
from gestures.gesture_types import GestureName
from gestures.recognizer import RuleBasedGestureRecognizer
from gestures.static_gestures import classify_static
from tests.helpers import (
    fist_hand,
    hand_at_wrist,
    index_point_hand,
    open_hand,
    pinch_hand,
    two_finger_hand,
)

PINCH_THRESHOLD = 0.05


def _classify(hand):
    features = FeatureExtractor().extract(hand)
    return classify_static(features, PINCH_THRESHOLD)


def test_open_hand():
    gesture, confidence = _classify(open_hand())
    assert gesture == GestureName.OPEN_HAND
    assert confidence > 0.6


def test_fist():
    gesture, confidence = _classify(fist_hand())
    assert gesture == GestureName.FIST
    assert confidence > 0.6


def test_index_point():
    gesture, confidence = _classify(index_point_hand())
    assert gesture == GestureName.INDEX_POINT
    assert confidence > 0.6


def test_two_finger():
    gesture, confidence = _classify(two_finger_hand())
    assert gesture == GestureName.TWO_FINGER
    assert confidence > 0.6


def test_pinch():
    gesture, confidence = _classify(pinch_hand())
    assert gesture == GestureName.PINCH
    assert confidence > 0.6


def test_pinch_takes_priority_even_with_folded_fingers():
    # PINCH should win even though the hand otherwise looks fist-like.
    gesture, _ = _classify(pinch_hand())
    assert gesture == GestureName.PINCH


def _run_translation(recognizer, extractor, start_x, end_x, steps=6, dt=0.05):
    gestures = []
    for i in range(steps):
        t = i * dt
        x = start_x + (end_x - start_x) * (i / (steps - 1))
        hand = hand_at_wrist((x, 0.9), pose="open")
        features = extractor.extract(hand, timestamp=t)
        result = recognizer.recognize(features)
        gestures.append(result.gesture)
    return gestures


def _make_recognizer() -> RuleBasedGestureRecognizer:
    return RuleBasedGestureRecognizer(
        pinch_threshold=PINCH_THRESHOLD,
        swipe_min_distance=0.15,
        swipe_max_duration=0.6,
        swipe_min_velocity=0.4,
        swipe_cooldown=0.5,
    )


def test_swipe_right_detected_on_fast_rightward_motion():
    gestures = _run_translation(_make_recognizer(), FeatureExtractor(), 0.3, 0.7)
    assert GestureName.SWIPE_RIGHT in gestures


def test_swipe_left_detected_on_fast_leftward_motion():
    gestures = _run_translation(_make_recognizer(), FeatureExtractor(), 0.7, 0.3)
    assert GestureName.SWIPE_LEFT in gestures


def test_no_swipe_on_slow_motion():
    # Same total distance, spread over a much longer time -> below min velocity.
    recognizer = _make_recognizer()
    extractor = FeatureExtractor()
    gestures = []
    steps = 6
    for i in range(steps):
        t = i * 2.0  # much slower than the 0.6s max_duration window
        x = 0.3 + (0.7 - 0.3) * (i / (steps - 1))
        hand = hand_at_wrist((x, 0.9), pose="open")
        features = extractor.extract(hand, timestamp=t)
        result = recognizer.recognize(features)
        gestures.append(result.gesture)
    assert GestureName.SWIPE_LEFT not in gestures
    assert GestureName.SWIPE_RIGHT not in gestures


def test_swipe_cooldown_prevents_double_trigger():
    recognizer = _make_recognizer()
    extractor = FeatureExtractor()
    gestures = _run_translation(recognizer, extractor, 0.3, 0.7, steps=6, dt=0.05)
    swipe_count = sum(1 for g in gestures if g == GestureName.SWIPE_RIGHT)
    assert swipe_count == 1
