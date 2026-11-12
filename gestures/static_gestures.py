"""Static gesture classification: pure functions that turn a single
``HandFeatures`` snapshot into a (gesture name, confidence) guess.

No temporal logic lives here — that's the state machine's job. This
module only answers "what does this one frame look like?".
"""

from __future__ import annotations

from gestures.features import HandFeatures
from gestures.gesture_types import GestureName


def classify_static(
    features: HandFeatures,
    pinch_threshold: float,
    was_pinching: bool = False,
    pinch_release_ratio: float = 1.3,
) -> tuple[GestureName, float]:
    """Classify a single frame's pose. Checked in priority order because
    some poses (PINCH) should win over an otherwise-ambiguous finger count.

    ``was_pinching`` / ``pinch_release_ratio`` add hysteresis: the caller
    (which owns the temporal state) says whether a pinch was already
    established, and if so the pinch is judged against a wider threshold
    so it doesn't flicker off at the boundary. With the defaults the
    behavior is identical to the original function.
    """
    ext = features.finger_extended
    count = features.extended_count

    # PINCH takes priority: a deliberate thumb-index touch should register
    # even if other fingers are in an ambiguous position.
    pinch_conf = pinch_confidence(
        features.thumb_index_distance,
        pinch_threshold,
        was_pinching,
        pinch_release_ratio,
    )
    if pinch_conf > 0:
        return GestureName.PINCH, pinch_conf

    if count == 0:
        return GestureName.FIST, 1.0

    if ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"]:
        conf = 0.9 if not ext["thumb"] else 0.75
        return GestureName.TWO_FINGER, conf

    if ext["index"] and not ext["middle"] and not ext["ring"] and not ext["pinky"]:
        conf = 0.9 if not ext["thumb"] else 0.8
        return GestureName.INDEX_POINT, conf

    if count >= 4:
        return GestureName.OPEN_HAND, min(1.0, count / 5.0)

    return GestureName.NONE, 0.0


def pinch_confidence(
    distance: float,
    pinch_threshold: float,
    was_pinching: bool = False,
    pinch_release_ratio: float = 1.3,
) -> float:
    """Public pinch confidence for a normalized thumb-index distance.

    When ``was_pinching`` is True the threshold is widened by
    ``pinch_release_ratio`` (hysteresis). Shared by the single-hand
    classifier and the two-hand pinch check so both judge a pinch the
    same way.
    """
    effective = pinch_threshold * pinch_release_ratio if was_pinching else pinch_threshold
    return _pinch_confidence(distance, effective)


def _pinch_confidence(distance: float, threshold: float) -> float:
    """Return 0 if clearly not pinched, otherwise a confidence in (0, 1]
    that increases as the fingertips get closer together."""
    if distance >= threshold:
        return 0.0
    if threshold <= 0:
        return 0.0
    closeness = 1.0 - (distance / threshold)
    return max(0.0, min(1.0, closeness))