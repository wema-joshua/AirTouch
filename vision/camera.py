"""Webcam capture, isolated behind a small interface.

Keeping OpenCV's VideoCapture wrapped here means the rest of the app
depends on ``Camera``, not on cv2 directly — useful for testing and for
swapping in a different capture backend later.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


class CameraError(Exception):
    """Raised when the webcam cannot be opened or read from."""


@dataclass(frozen=True)
class Frame:
    """A single captured frame."""

    image: np.ndarray  # BGR, as produced by OpenCV
    width: int
    height: int
    timestamp: float


class Camera:
    """Thin wrapper around cv2.VideoCapture with explicit lifecycle."""

    def __init__(
        self,
        device: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        mirrored: bool = True,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.mirrored = mirrored
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self._cap is not None:
            return
        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            raise CameraError(
                f"Could not open webcam (device={self.device}). "
                "Check that it is connected and not in use by another "
                "application, and that camera permissions are granted."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap = cap
        logger.info(
            "Camera opened (device=%s, requested=%sx%s@%sfps)",
            self.device,
            self.width,
            self.height,
            self.fps,
        )

    def read(self) -> Frame | None:
        """Read one frame. Returns None if a frame could not be captured
        (transient failures are tolerated by callers, not raised here)."""
        if self._cap is None:
            raise CameraError("Camera.read() called before Camera.open()")
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        if self.mirrored:
            image = cv2.flip(image, 1)
        h, w = image.shape[:2]
        return Frame(image=image, width=w, height=h, timestamp=cv2.getTickCount() / cv2.getTickFrequency())

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
