"""Draws the debug/HUD overlay onto a frame using OpenCV primitives.

Pure rendering: takes already-computed values (gesture name, mode, fps,
etc.) and draws them. It never computes gesture logic itself.
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from vision.landmarks import TrackedHands

_TEXT_COLOR = (255, 255, 255)
_ACCENT_COLOR = (0, 220, 120)
_PAUSED_COLOR = (0, 165, 255)
_WARN_COLOR = (60, 60, 255)
_SKELETON_COLOR = (0, 220, 120)
_LANDMARK_COLOR = (255, 200, 0)


def draw_skeleton(
    image: np.ndarray,
    tracked_hands: TrackedHands,
    connections: Iterable[tuple[int, int]],
    show_landmarks: bool = True,
) -> None:
    if not tracked_hands.detected:
        return
    h, w = image.shape[:2]
    for hand in tracked_hands.hands:
        points_px = [(int(p.x * w), int(p.y * h)) for p in hand.landmarks]
        for a, b in connections:
            cv2.line(image, points_px[a], points_px[b], _SKELETON_COLOR, 2)
        if show_landmarks:
            for px, py in points_px:
                cv2.circle(image, (px, py), 4, _LANDMARK_COLOR, -1)


def draw_hud(
    image: np.ndarray,
    *,
    fps: float,
    hand_detected: bool,
    gesture: str,
    mode: str,
    debug_lines: list[str] | None = None,
) -> None:
    mode_color = _ACCENT_COLOR
    if mode == "PAUSED":
        mode_color = _PAUSED_COLOR
    elif mode == "STOPPING":
        mode_color = _WARN_COLOR

    cv2.putText(image, "AIRTOUCH", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, _ACCENT_COLOR, 2)

    status = "DETECTED" if hand_detected else "NOT DETECTED"
    status_color = _ACCENT_COLOR if hand_detected else _WARN_COLOR
    lines = [
        (f"Hand: {status}", status_color),
        (f"Gesture: {gesture}", _TEXT_COLOR),
        (f"Mode: {mode}", mode_color),
        (f"FPS: {fps:.1f}", _TEXT_COLOR),
    ]

    y = 66
    for text, color in lines:
        cv2.putText(image, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 26

    if debug_lines:
        y += 6
        for line in debug_lines:
            cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 20

    cv2.putText(
        image,
        "Q / ESC: quit",
        (16, image.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )
