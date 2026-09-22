"""Pure geometry helpers operating on plain (x, y[, z]) tuples.

These functions have no dependency on MediaPipe or OpenCV types so they
can be unit tested with synthetic data and reused by any future vision
backend (e.g. a different hand-tracking library).
"""

from __future__ import annotations

import math
from typing import Sequence

Point = Sequence[float]  # (x, y) or (x, y, z)


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2D or 3D points."""
    return math.dist(tuple(a), tuple(b))


def midpoint(a: Point, b: Point) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def angle_between(a: Point, b: Point, c: Point) -> float:
    """Angle in degrees at vertex ``b`` formed by rays b->a and b->c."""
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    mag_a = math.hypot(ax, ay)
    mag_c = math.hypot(cx, cy)
    if mag_a == 0 or mag_c == 0:
        return 0.0
    cos_theta = (ax * cx + ay * cy) / (mag_a * mag_c)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def bounding_box(points: Sequence[Point]) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) for a set of points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t
