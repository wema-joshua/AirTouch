"""Synthetic hand builders for testing the gesture engine without a webcam.

Per the project's testability requirement, the recognizer accepts
processed features rather than requiring a camera — these helpers build
plausible, deterministic 21-landmark hands for each canonical pose so
tests can validate feature extraction and gesture classification
directly.

Coordinates are hand-tuned normalized (0..1) values, verified against
``FeatureExtractor``/``classify_static`` to produce the intended pose.
"""

from __future__ import annotations

from vision.landmarks import Hand, Landmark, LandmarkPoint

WRIST = (0.5, 0.9)

_FINGER_MCP_POS = {
    "index": (0.425, 0.75),
    "middle": (0.5, 0.75),
    "ring": (0.575, 0.75),
    "pinky": (0.65, 0.76),
}

_FINGER_LANDMARKS = {
    "index": (Landmark.INDEX_MCP, Landmark.INDEX_PIP, Landmark.INDEX_DIP, Landmark.INDEX_TIP),
    "middle": (Landmark.MIDDLE_MCP, Landmark.MIDDLE_PIP, Landmark.MIDDLE_DIP, Landmark.MIDDLE_TIP),
    "ring": (Landmark.RING_MCP, Landmark.RING_PIP, Landmark.RING_DIP, Landmark.RING_TIP),
    "pinky": (Landmark.PINKY_MCP, Landmark.PINKY_PIP, Landmark.PINKY_DIP, Landmark.PINKY_TIP),
}

# Default tip positions for a fully folded (fist-like) hand.
_FOLDED_TIPS = {
    "index": (0.425, 0.78),
    "middle": (0.5, 0.78),
    "ring": (0.575, 0.78),
    "pinky": (0.65, 0.79),
}

# Default tip positions for a fully extended (open) hand.
_EXTENDED_TIPS = {
    "index": (0.425, 0.30),
    "middle": (0.5, 0.28),
    "ring": (0.575, 0.30),
    "pinky": (0.65, 0.33),
}

_THUMB_TIP_EXTENDED = (0.25, 0.65)
_THUMB_TIP_FOLDED = (0.45, 0.78)


def build_hand(
    finger_tips: dict[str, tuple[float, float]] | None = None,
    thumb_tip: tuple[float, float] = _THUMB_TIP_FOLDED,
    wrist: tuple[float, float] = WRIST,
    handedness: str = "Right",
    score: float = 0.95,
) -> Hand:
    """Build a 21-landmark Hand. ``finger_tips`` overrides index/middle/
    ring/pinky tip positions (defaults to fully folded); PIP/DIP joints
    are interpolated toward the tip so extension detection behaves
    sensibly for in-between poses too.
    """
    tips = dict(_FOLDED_TIPS)
    if finger_tips:
        tips.update(finger_tips)

    points: list[tuple[float, float] | None] = [None] * 21
    points[Landmark.WRIST] = wrist

    thumb_mcp = (0.45, 0.8)
    thumb_ip = ((thumb_mcp[0] + thumb_tip[0]) / 2, (thumb_mcp[1] + thumb_tip[1]) / 2)
    points[Landmark.THUMB_CMC] = (0.47, 0.85)
    points[Landmark.THUMB_MCP] = thumb_mcp
    points[Landmark.THUMB_IP] = thumb_ip
    points[Landmark.THUMB_TIP] = thumb_tip

    for name, (mcp_i, pip_i, dip_i, tip_i) in _FINGER_LANDMARKS.items():
        mcp_pos = _FINGER_MCP_POS[name]
        tip = tips[name]
        pip = ((mcp_pos[0] + tip[0]) / 2, (mcp_pos[1] + tip[1]) * 0.5 + 0.05)
        dip = ((mcp_pos[0] + tip[0]) * 0.5, (mcp_pos[1] + tip[1]) * 0.5)
        points[mcp_i] = mcp_pos
        points[pip_i] = pip
        points[dip_i] = dip
        points[tip_i] = tip

    landmark_points = tuple(LandmarkPoint(x=p[0], y=p[1], z=0.0) for p in points)
    return Hand(landmarks=landmark_points, handedness=handedness, score=score)


def open_hand() -> Hand:
    return build_hand(finger_tips=_EXTENDED_TIPS, thumb_tip=_THUMB_TIP_EXTENDED)


def fist_hand() -> Hand:
    return build_hand(finger_tips=_FOLDED_TIPS, thumb_tip=_THUMB_TIP_FOLDED)


def index_point_hand() -> Hand:
    tips = dict(_FOLDED_TIPS)
    tips["index"] = _EXTENDED_TIPS["index"]
    return build_hand(finger_tips=tips, thumb_tip=_THUMB_TIP_FOLDED)


def two_finger_hand() -> Hand:
    tips = dict(_FOLDED_TIPS)
    tips["index"] = _EXTENDED_TIPS["index"]
    tips["middle"] = _EXTENDED_TIPS["middle"]
    return build_hand(finger_tips=tips, thumb_tip=_THUMB_TIP_FOLDED)


def pinch_hand(gap: float = 0.001) -> Hand:
    index_tip = (0.42, 0.6)
    thumb_tip = (index_tip[0] + gap, index_tip[1] + gap)
    tips = dict(_FOLDED_TIPS)
    tips["index"] = index_tip
    return build_hand(finger_tips=tips, thumb_tip=thumb_tip)


def hand_at_wrist(wrist: tuple[float, float], pose: str = "index_point") -> Hand:
    """A hand in a given pose, translated so its wrist sits at ``wrist``.
    Useful for building a sequence of frames that move across the screen
    (e.g. for swipe/velocity tests)."""
    dx = wrist[0] - WRIST[0]
    dy = wrist[1] - WRIST[1]

    base = {
        "open": open_hand(),
        "fist": fist_hand(),
        "index_point": index_point_hand(),
        "two_finger": two_finger_hand(),
        "pinch": pinch_hand(),
    }[pose]

    shifted = tuple(
        LandmarkPoint(x=p.x + dx, y=p.y + dy, z=p.z) for p in base.landmarks
    )
    return Hand(landmarks=shifted, handedness=base.handedness, score=base.score)
