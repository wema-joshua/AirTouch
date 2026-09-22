"""Gesture recognizer interface, plus the V1 rule-based implementation.

``GestureRecognizer`` is the seam where a future ``MLGestureRecognizer``
or ``HybridGestureRecognizer`` plugs in: both would accept the same
``HandFeatures`` and return the same ``GestureResult`` shape, so nothing
upstream (features) or downstream (state machine, actions) needs to
change when a learned model replaces or augments the rules below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from gestures.features import HandFeatures
from gestures.gesture_types import GestureName, GestureResult
from gestures.motion_gestures import SwipeDetector
from gestures.static_gestures import classify_static


class GestureRecognizer(ABC):
    """Common interface for anything that turns features into a gesture guess."""

    @abstractmethod
    def recognize(self, features: HandFeatures) -> GestureResult:
        """Classify a single frame's features. Implementations may keep
        internal state (e.g. motion buffers) across calls."""

    def reset(self) -> None:
        """Clear any internal temporal state (e.g. on tracking loss)."""


class RuleBasedGestureRecognizer(GestureRecognizer):
    """V1 recognizer: hand-written rules for static poses, plus a swipe
    detector for motion. Swipe results take priority over static poses
    for a single frame since they represent a completed, discrete event.
    """

    def __init__(
        self,
        pinch_threshold: float,
        swipe_min_distance: float,
        swipe_max_duration: float,
        swipe_min_velocity: float,
        swipe_cooldown: float,
    ) -> None:
        self.pinch_threshold = pinch_threshold
        self._swipe_detector = SwipeDetector(
            min_distance=swipe_min_distance,
            max_duration=swipe_max_duration,
            min_velocity=swipe_min_velocity,
            cooldown=swipe_cooldown,
        )

    def reset(self) -> None:
        self._swipe_detector.reset()

    def recognize(self, features: HandFeatures) -> GestureResult:
        index_tip = features.points[8]  # Landmark.INDEX_TIP

        swipe = self._swipe_detector.update(features.timestamp, features.palm_center)
        if swipe is not None:
            return GestureResult(
                gesture=swipe,
                confidence=1.0,
                palm_center=features.palm_center,
                index_tip=index_tip,
                velocity=features.velocity,
                thumb_index_distance=features.thumb_index_distance,
                timestamp=features.timestamp,
            )

        gesture, confidence = classify_static(features, self.pinch_threshold)
        return GestureResult(
            gesture=gesture,
            confidence=confidence,
            palm_center=features.palm_center,
            index_tip=index_tip,
            velocity=features.velocity,
            thumb_index_distance=features.thumb_index_distance,
            timestamp=features.timestamp,
        )
