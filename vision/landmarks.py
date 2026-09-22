"""Hand landmark data structures.

These are plain, framework-agnostic dataclasses. The vision layer is
responsible for translating MediaPipe's output into these types; nothing
downstream (features, gestures, actions) should ever import mediapipe.

Even though V1 only acts on a single hand, ``TrackedHands`` holds a list
so the two-hand gesture engine described in the roadmap can be added
without changing this module's public shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Landmark(IntEnum):
    """The 21 standard MediaPipe hand landmark indices, named for readability."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


NUM_LANDMARKS = 21


@dataclass(frozen=True)
class LandmarkPoint:
    """A single landmark in normalized [0, 1] image coordinates, plus
    MediaPipe's relative depth (z, roughly camera-relative, not metric)."""

    x: float
    y: float
    z: float = 0.0

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def as_tuple3(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class Hand:
    """A single detected hand: 21 normalized landmarks plus handedness."""

    landmarks: tuple[LandmarkPoint, ...]  # length == NUM_LANDMARKS
    handedness: str  # "Left" or "Right", as reported by the tracker
    score: float  # detection confidence [0, 1]

    def point(self, landmark: "Landmark | int") -> LandmarkPoint:
        return self.landmarks[int(landmark)]

    def __post_init__(self) -> None:
        if len(self.landmarks) != NUM_LANDMARKS:
            raise ValueError(
                f"Hand requires exactly {NUM_LANDMARKS} landmarks, "
                f"got {len(self.landmarks)}"
            )


@dataclass(frozen=True)
class TrackedHands:
    """Result of processing a single frame: zero or more detected hands.

    V1's pipeline consumes only the first (primary) hand via
    ``.primary``, but the list shape is what future two-hand gesture
    support needs — no restructuring required later.
    """

    hands: tuple[Hand, ...]
    frame_width: int
    frame_height: int
    timestamp: float

    @property
    def primary(self) -> Hand | None:
        return self.hands[0] if self.hands else None

    @property
    def detected(self) -> bool:
        return len(self.hands) > 0
