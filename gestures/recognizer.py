"""Gesture recognizer interface, plus the V1 rule-based implementation.

``GestureRecognizer`` is the seam where a future ``MLGestureRecognizer``
or ``HybridGestureRecognizer`` plugs in: both would accept the same
``HandFeatures`` and return the same ``GestureResult`` shape, so nothing
upstream (features) or downstream (state machine, actions) needs to
change when a learned model replaces or augments the rules below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from gestures.features import HandFeatures, TwoHandFeatures
from gestures.gesture_types import GestureName, GestureResult
from gestures.motion_gestures import DeltaTracker, ScalarDeltaTracker, SwipeDetector
from gestures.static_gestures import classify_static, pinch_confidence


class GestureRecognizer(ABC):
    """Common interface for anything that turns features into a gesture guess."""

    @abstractmethod
    def recognize(self, features: HandFeatures) -> GestureResult:
        """Classify a single frame's features. Implementations may keep
        internal state (e.g. motion buffers) across calls."""

    def recognize_pair(self, pair: TwoHandFeatures) -> GestureResult:
        """Classify a frame in which two hands are visible. The default
        recognizes no two-hand gestures; recognizers that support them
        override this."""
        return GestureResult(
            gesture=GestureName.NONE,
            confidence=0.0,
            timestamp=pair.timestamp,
        )

    def reset(self) -> None:
        """Clear any internal temporal state (e.g. on tracking loss)."""


class RuleBasedGestureRecognizer(GestureRecognizer):
    """V1 recognizer: hand-written rules for static poses, plus a swipe
    detector for motion. Swipe results take priority over static poses
    for a single frame since they represent a completed, discrete event.

    For 3D control it also fills in smoothed per-frame deltas while a
    pinch is held (``GestureResult.delta``), and recognizes
    ``TWO_HAND_PINCH`` via ``recognize_pair``.

    Set ``suppress_swipe_while_pinched`` to True in 3D mode so fast
    orbit/pan movement while pinching isn't mistaken for a swipe.

    ``min_pinch_confidence`` must equal the state machine's
    ``min_confidence``: the state machine ignores any gesture below that
    confidence, so pinch hysteresis has to latch at the same point.
    """

    def __init__(
        self,
        pinch_threshold: float,
        swipe_min_distance: float,
        swipe_max_duration: float,
        swipe_min_velocity: float,
        swipe_cooldown: float,
        pinch_release_ratio: float = 1.3,
        orbit_deadzone: float = 0.001,
        pair_deadzone: float = 0.002,
        suppress_swipe_while_pinched: bool = False,
        min_pinch_confidence: float = 0.6,
    ) -> None:
        self.pinch_threshold = pinch_threshold
        self.pinch_release_ratio = pinch_release_ratio
        self.min_pinch_confidence = min_pinch_confidence
        self.suppress_swipe_while_pinched = suppress_swipe_while_pinched
        self._swipe_detector = SwipeDetector(
            min_distance=swipe_min_distance,
            max_duration=swipe_max_duration,
            min_velocity=swipe_min_velocity,
            cooldown=swipe_cooldown,
        )
        self._orbit = DeltaTracker(deadzone=orbit_deadzone)
        self._pair_distance = ScalarDeltaTracker(deadzone=pair_deadzone)
        self._pair_midpoint = DeltaTracker(deadzone=pair_deadzone)
        self._was_pinching = False
        self._pinch_latched = False
        self._pair_engaged = False
        self._pair_latched = False

    def reset(self) -> None:
        self._swipe_detector.reset()
        self._orbit.reset()
        self._pair_distance.reset()
        self._pair_midpoint.reset()
        self._was_pinching = False
        self._pinch_latched = False
        self._pair_engaged = False
        self._pair_latched = False

    def recognize(self, features: HandFeatures) -> GestureResult:
        index_tip = features.points[8]  # Landmark.INDEX_TIP

        if self.suppress_swipe_while_pinched and self._was_pinching:
            self._swipe_detector.reset()
            swipe = None
        else:
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

        gesture, confidence = classify_static(
            features,
            self.pinch_threshold,
            was_pinching=self._pinch_latched,
            pinch_release_ratio=self.pinch_release_ratio,
        )

        pinching = gesture in (GestureName.PINCH, GestureName.PINCH_HOLD)
        if pinching:
            delta = self._orbit.update(features.timestamp, features.palm_center)
        else:
            self._orbit.reset()
            delta = (0.0, 0.0)
        self._was_pinching = pinching
        self._pinch_latched = (
            gesture == GestureName.PINCH and confidence >= self.min_pinch_confidence
        )

        return GestureResult(
            gesture=gesture,
            confidence=confidence,
            palm_center=features.palm_center,
            index_tip=index_tip,
            velocity=features.velocity,
            thumb_index_distance=features.thumb_index_distance,
            timestamp=features.timestamp,
            delta=delta,
        )

    def recognize_pair(self, pair: TwoHandFeatures) -> GestureResult:
        """TWO_HAND_PINCH while both hands are pinched.

        Each hand's pinch is judged exactly like the single-hand pinch
        (same confidence function, same hysteresis), and the gesture's
        confidence is the weaker of the two. That way the state machine's
        ``min_confidence`` gate treats one-hand and two-hand pinches
        consistently.

        The first engaged frame reports zero deltas so nothing jumps.
        """
        conf_left = pinch_confidence(
            pair.left.thumb_index_distance,
            self.pinch_threshold,
            self._pair_latched,
            self.pinch_release_ratio,
        )
        conf_right = pinch_confidence(
            pair.right.thumb_index_distance,
            self.pinch_threshold,
            self._pair_latched,
            self.pinch_release_ratio,
        )
        confidence = min(conf_left, conf_right)

        if confidence <= 0.0:
            self._pair_distance.reset()
            self._pair_midpoint.reset()
            self._pair_engaged = False
            self._pair_latched = False
            return GestureResult(
                gesture=GestureName.NONE,
                confidence=0.0,
                palm_center=pair.midpoint,
                timestamp=pair.timestamp,
            )

        # Two-hand control takes over from single-hand orbit tracking.
        self._orbit.reset()

        distance_delta = self._pair_distance.update(pair.timestamp, pair.hand_distance)
        midpoint_delta = self._pair_midpoint.update(pair.timestamp, pair.midpoint)
        angle_delta = pair.angle_delta_deg if self._pair_engaged else 0.0
        self._pair_engaged = True
        self._pair_latched = confidence >= self.min_pinch_confidence

        return GestureResult(
            gesture=GestureName.TWO_HAND_PINCH,
            confidence=confidence,
            palm_center=pair.midpoint,
            thumb_index_distance=(
                pair.left.thumb_index_distance + pair.right.thumb_index_distance
            )
            / 2.0,
            timestamp=pair.timestamp,
            pair_distance_delta=distance_delta,
            pair_midpoint_delta=midpoint_delta,
            pair_angle_delta_deg=angle_delta,
        )