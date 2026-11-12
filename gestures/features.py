"""Feature extraction: turns a raw ``Hand`` into normalized, reusable
numeric features.

This is the layer both the rule-based recognizer AND a future ML
recognizer are meant to consume. Keeping it separate from gesture logic
means adding MLGestureRecognizer later requires no changes here.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from utils.geometry import angle_between, bounding_box, distance, midpoint
from vision.landmarks import Hand, Landmark

# Fingertip / MCP landmark groupings used for extension detection.
_FINGER_TIPS = {
    "thumb": Landmark.THUMB_TIP,
    "index": Landmark.INDEX_TIP,
    "middle": Landmark.MIDDLE_TIP,
    "ring": Landmark.RING_TIP,
    "pinky": Landmark.PINKY_TIP,
}
_FINGER_PIPS = {
    "thumb": Landmark.THUMB_IP,
    "index": Landmark.INDEX_PIP,
    "middle": Landmark.MIDDLE_PIP,
    "ring": Landmark.RING_PIP,
    "pinky": Landmark.PINKY_PIP,
}
_FINGER_MCPS = {
    "thumb": Landmark.THUMB_MCP,
    "index": Landmark.INDEX_MCP,
    "middle": Landmark.MIDDLE_MCP,
    "ring": Landmark.RING_MCP,
    "pinky": Landmark.PINKY_MCP,
}


class FingerName(str, Enum):
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    PINKY = "pinky"


@dataclass(frozen=True)
class HandFeatures:
    """A snapshot of derived, normalized measurements for one hand at one
    point in time. All distances are in normalized [0, 1] image-space
    units (as reported by MediaPipe), which keeps them roughly stable
    across different distances from the camera since they're relative
    to the hand's own landmarks and the frame, not absolute pixels.
    """

    timestamp: float
    handedness: str
    detection_score: float

    # Raw-ish, normalized landmark positions (x, y), indexable by Landmark.
    points: tuple[tuple[float, float], ...]

    # Per-finger boolean: True if extended (straightened).
    finger_extended: dict[str, bool]

    # Per-finger PIP joint angle in degrees (proxy for curl).
    finger_angles: dict[str, float]

    # Key distances.
    thumb_index_distance: float
    palm_center: tuple[float, float]
    hand_span: float  # distance from wrist to middle MCP, used to normalize other distances

    # Whole-hand descriptors.
    orientation_deg: float  # angle of wrist->middle_mcp vector, 0 = pointing right
    bounding_box: tuple[float, float, float, float]  # min_x, min_y, max_x, max_y

    # Motion, relative to the previous frame this hand was seen (0 if unknown).
    velocity: tuple[float, float]  # normalized units / second
    speed: float
    acceleration: tuple[float, float]  # normalized units / second^2

    @property
    def extended_count(self) -> int:
        return sum(1 for v in self.finger_extended.values() if v)


class FeatureExtractor:
    """Stateful extractor that also computes velocity/acceleration by
    comparing against the previous frame. Bounded history (2 samples)
    keeps memory use constant regardless of session length.
    """

    def __init__(self) -> None:
        self._history: deque[tuple[float, tuple[float, float]]] = deque(maxlen=2)

    def reset(self) -> None:
        self._history.clear()

    def extract(self, hand: Hand, timestamp: float | None = None) -> HandFeatures:
        ts = timestamp if timestamp is not None else time.monotonic()
        points = tuple(p.as_tuple() for p in hand.landmarks)

        wrist = points[Landmark.WRIST]
        middle_mcp = points[Landmark.MIDDLE_MCP]
        hand_span = max(distance(wrist, middle_mcp), 1e-6)

        finger_extended = self._compute_extension(points, hand.handedness)
        finger_angles = self._compute_finger_angles(points)

        thumb_index_distance = distance(
            points[Landmark.THUMB_TIP], points[Landmark.INDEX_TIP]
        ) / hand_span

        palm_points = [
            points[i]
            for i in (
                Landmark.WRIST,
                Landmark.INDEX_MCP,
                Landmark.MIDDLE_MCP,
                Landmark.RING_MCP,
                Landmark.PINKY_MCP,
            )
        ]
        palm_center = (
            sum(p[0] for p in palm_points) / len(palm_points),
            sum(p[1] for p in palm_points) / len(palm_points),
        )

        orientation_deg = _vector_angle_deg(wrist, middle_mcp)
        bbox = bounding_box(points)

        velocity, acceleration = self._compute_motion(palm_center, ts)
        speed = distance((0.0, 0.0), velocity)

        return HandFeatures(
            timestamp=ts,
            handedness=hand.handedness,
            detection_score=hand.score,
            points=points,
            finger_extended={k: v for k, v in finger_extended.items()},
            finger_angles=finger_angles,
            thumb_index_distance=thumb_index_distance,
            palm_center=palm_center,
            hand_span=hand_span,
            orientation_deg=orientation_deg,
            bounding_box=bbox,
            velocity=velocity,
            speed=speed,
            acceleration=acceleration,
        )

    def _compute_motion(
        self, palm_center: tuple[float, float], ts: float
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        prev_velocity = (0.0, 0.0)
        velocity = (0.0, 0.0)
        acceleration = (0.0, 0.0)

        if self._history:
            prev_ts, prev_center = self._history[-1]
            dt = max(ts - prev_ts, 1e-6)
            velocity = (
                (palm_center[0] - prev_center[0]) / dt,
                (palm_center[1] - prev_center[1]) / dt,
            )
            if len(self._history) == 2:
                prev_prev_ts, _ = self._history[0]
                prev_dt = max(prev_ts - prev_prev_ts, 1e-6)
                # Approximate previous velocity using the two oldest samples.
                _, prev_prev_center = self._history[0]
                prev_velocity = (
                    (prev_center[0] - prev_prev_center[0]) / prev_dt,
                    (prev_center[1] - prev_prev_center[1]) / prev_dt,
                )
                acceleration = (
                    (velocity[0] - prev_velocity[0]) / dt,
                    (velocity[1] - prev_velocity[1]) / dt,
                )

        self._history.append((ts, palm_center))
        return velocity, acceleration

    @staticmethod
    def _compute_extension(
        points: tuple[tuple[float, float], ...], handedness: str
    ) -> dict[str, bool]:
        result: dict[str, bool] = {}
        wrist = points[Landmark.WRIST]
        middle_mcp = points[Landmark.MIDDLE_MCP]
        hand_span = max(distance(wrist, middle_mcp), 1e-6)

        for name in FingerName:
            tip = points[_FINGER_TIPS[name.value]]
            mcp = points[_FINGER_MCPS[name.value]]
            if name is FingerName.THUMB:
                # Thumb extension is better judged by distance from the palm
                # (index MCP) than by y-position, since it moves sideways.
                ref = points[Landmark.INDEX_MCP]
                extended = distance(tip, ref) / hand_span > 0.6
            else:
                # A finger is "extended" if its tip is meaningfully farther
                # from the wrist than its MCP joint is (straightened out),
                # which works regardless of hand rotation.
                tip_dist = distance(tip, wrist)
                mcp_dist = distance(mcp, wrist)
                extended = tip_dist > mcp_dist * 1.15
            result[name.value] = extended
        return result

    @staticmethod
    def _compute_finger_angles(
        points: tuple[tuple[float, float], ...]
    ) -> dict[str, float]:
        angles: dict[str, float] = {}
        for name in FingerName:
            mcp = points[_FINGER_MCPS[name.value]]
            pip = points[_FINGER_PIPS[name.value]]
            tip = points[_FINGER_TIPS[name.value]]
            angles[name.value] = angle_between(mcp, pip, tip)
        return angles


def _vector_angle_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    import math

    dx, dy = b[0] - a[0], b[1] - a[1]
    return math.degrees(math.atan2(dy, dx))


# --- Two-hand features ---------------------------------------------------


@dataclass(frozen=True)
class TwoHandFeatures:
    """Relationship between two hands visible in the same frame.

    ``left`` / ``right`` refer to image position (leftmost hand in the
    frame is ``left``), not MediaPipe's handedness label, which can be
    unreliable or mirrored depending on camera flipping.

    ``*_delta`` fields are changes since the previous frame in which
    both hands were visible (0 on the first frame, so a zoom or rotate
    never jumps when the second hand appears).
    """

    timestamp: float
    left: HandFeatures
    right: HandFeatures

    # Palm-center to palm-center distance, divided by the mean hand_span of
    # both hands so it stays comparable at different distances from the camera.
    hand_distance: float
    distance_delta: float  # change in hand_distance since last frame (zoom)
    distance_rate: float  # hand_distance change per second

    # Midpoint between the two palm centers (normalized image units).
    midpoint: tuple[float, float]
    midpoint_delta: tuple[float, float]  # frame-to-frame change (pan)

    # Angle of the left->right palm vector, 0 = level, in degrees.
    angle_deg: float
    angle_delta_deg: float  # wrapped to [-180, 180] (twist / roll)


@dataclass(frozen=True)
class MultiHandFeatures:
    """Everything extracted from one frame, for 0, 1 or 2 hands."""

    timestamp: float
    hands: dict[str, HandFeatures]  # keyed by "left" / "right" (image position)
    pair: TwoHandFeatures | None  # only set when exactly two hands are visible


class MultiHandFeatureExtractor:
    """Runs one ``FeatureExtractor`` per hand and, when two hands are
    visible, derives the relationship features between them.

    Each hand gets its own ``FeatureExtractor`` because that class keeps a
    per-hand motion history; sharing one would corrupt velocity values.
    Whenever the number of visible hands changes, all histories are reset
    so slot reassignment can't produce a velocity or delta spike.
    """

    def __init__(self) -> None:
        self._extractors: dict[str, FeatureExtractor] = {
            "left": FeatureExtractor(),
            "right": FeatureExtractor(),
        }
        self._last_count = 0
        # (distance, midpoint, angle_deg, timestamp) from the previous pair frame.
        self._prev_pair: tuple[float, tuple[float, float], float, float] | None = None

    def reset(self) -> None:
        for extractor in self._extractors.values():
            extractor.reset()
        self._prev_pair = None

    def extract(
        self, hands: Sequence[Hand], timestamp: float | None = None
    ) -> MultiHandFeatures:
        ts = timestamp if timestamp is not None else time.monotonic()
        hands = list(hands)[:2]

        if len(hands) != self._last_count:
            self.reset()
            self._last_count = len(hands)

        if not hands:
            return MultiHandFeatures(timestamp=ts, hands={}, pair=None)

        if len(hands) == 2:
            ordered = sorted(hands, key=_wrist_x)
            slots = ("left", "right")
        else:
            label = hands[0].handedness.lower()
            slots = (label if label in self._extractors else "right",)
            ordered = hands

        per_hand = {
            slot: self._extractors[slot].extract(hand, ts)
            for slot, hand in zip(slots, ordered)
        }

        pair = self._compute_pair(per_hand, ts) if len(per_hand) == 2 else None
        return MultiHandFeatures(timestamp=ts, hands=per_hand, pair=pair)

    def _compute_pair(
        self, per_hand: dict[str, HandFeatures], ts: float
    ) -> TwoHandFeatures:
        left, right = per_hand["left"], per_hand["right"]

        mean_span = max((left.hand_span + right.hand_span) / 2.0, 1e-6)
        hand_distance = distance(left.palm_center, right.palm_center) / mean_span
        mid = (
            (left.palm_center[0] + right.palm_center[0]) / 2.0,
            (left.palm_center[1] + right.palm_center[1]) / 2.0,
        )
        angle = _vector_angle_deg(left.palm_center, right.palm_center)

        distance_delta = 0.0
        distance_rate = 0.0
        midpoint_delta = (0.0, 0.0)
        angle_delta = 0.0

        if self._prev_pair is not None:
            prev_dist, prev_mid, prev_angle, prev_ts = self._prev_pair
            dt = max(ts - prev_ts, 1e-6)
            distance_delta = hand_distance - prev_dist
            distance_rate = distance_delta / dt
            midpoint_delta = (mid[0] - prev_mid[0], mid[1] - prev_mid[1])
            angle_delta = _wrap_deg(angle - prev_angle)

        self._prev_pair = (hand_distance, mid, angle, ts)

        return TwoHandFeatures(
            timestamp=ts,
            left=left,
            right=right,
            hand_distance=hand_distance,
            distance_delta=distance_delta,
            distance_rate=distance_rate,
            midpoint=mid,
            midpoint_delta=midpoint_delta,
            angle_deg=angle,
            angle_delta_deg=angle_delta,
        )


def _wrist_x(hand: Hand) -> float:
    return hand.landmarks[Landmark.WRIST].as_tuple()[0]


def _wrap_deg(angle: float) -> float:
    """Wrap an angle difference into [-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0