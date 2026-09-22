"""Tests for gestures.features — synthetic landmark data, no webcam needed."""

from __future__ import annotations

from gestures.features import FeatureExtractor
from tests.helpers import fist_hand, index_point_hand, open_hand, pinch_hand, two_finger_hand


def test_open_hand_all_fingers_extended():
    extractor = FeatureExtractor()
    features = extractor.extract(open_hand())
    assert all(features.finger_extended.values())
    assert features.extended_count == 5


def test_fist_no_fingers_extended():
    extractor = FeatureExtractor()
    features = extractor.extract(fist_hand())
    assert not any(features.finger_extended.values())
    assert features.extended_count == 0


def test_index_point_only_index_extended():
    extractor = FeatureExtractor()
    features = extractor.extract(index_point_hand())
    assert features.finger_extended["index"] is True
    assert features.finger_extended["middle"] is False
    assert features.finger_extended["ring"] is False
    assert features.finger_extended["pinky"] is False


def test_two_finger_index_and_middle_extended():
    extractor = FeatureExtractor()
    features = extractor.extract(two_finger_hand())
    assert features.finger_extended["index"] is True
    assert features.finger_extended["middle"] is True
    assert features.finger_extended["ring"] is False
    assert features.finger_extended["pinky"] is False


def test_pinch_thumb_index_distance_is_small():
    extractor = FeatureExtractor()
    features = extractor.extract(pinch_hand())
    assert features.thumb_index_distance < 0.05


def test_open_hand_thumb_index_distance_is_large():
    extractor = FeatureExtractor()
    features = extractor.extract(open_hand())
    assert features.thumb_index_distance > 0.05


def test_velocity_is_zero_on_first_frame():
    extractor = FeatureExtractor()
    features = extractor.extract(open_hand(), timestamp=0.0)
    assert features.velocity == (0.0, 0.0)


def test_velocity_reflects_movement_between_frames():
    extractor = FeatureExtractor()
    extractor.extract(open_hand(), timestamp=0.0)
    # Build a hand with the same pose but shifted palm position.
    from tests.helpers import hand_at_wrist

    moved = hand_at_wrist((0.6, 0.9), pose="open")
    features = extractor.extract(moved, timestamp=1.0)
    # Moved to the right over 1 second -> positive x velocity.
    assert features.velocity[0] > 0


def test_reset_clears_motion_history():
    extractor = FeatureExtractor()
    extractor.extract(open_hand(), timestamp=0.0)
    extractor.reset()
    features = extractor.extract(open_hand(), timestamp=5.0)
    assert features.velocity == (0.0, 0.0)


def test_hand_span_is_positive():
    extractor = FeatureExtractor()
    features = extractor.extract(open_hand())
    assert features.hand_span > 0
