"""MediaPipe-backed hand landmark detector.

This is the ONLY module in the codebase allowed to import mediapipe.
It translates MediaPipe's output into the framework-agnostic
``TrackedHands`` / ``Hand`` / ``LandmarkPoint`` types defined in
``vision.landmarks`` so every downstream layer stays decoupled from the
specific tracking library. Swapping MediaPipe for another hand-tracking
backend later means rewriting only this file.
"""

from __future__ import annotations

import time

import mediapipe as mp
import numpy as np

from utils.logging import get_logger
from vision.landmarks import Hand, LandmarkPoint, NUM_LANDMARKS, TrackedHands

logger = get_logger(__name__)


class HandTrackerError(Exception):
    """Raised when the underlying MediaPipe model fails to initialize."""


class HandTracker:
    """Wraps MediaPipe HandLandmarker and exposes framework-agnostic results."""

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        max_num_hands: int = 1,
    ) -> None:
        try:
            base_options = mp.tasks.BaseOptions
            hand_landmarker = mp.tasks.vision.HandLandmarker
            hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions
            running_mode = mp.tasks.vision.RunningMode

            options = hand_landmarker_options(
                base_options=base_options(
                    model_asset_path="models/hand_landmarker.task"
                ),
                running_mode=running_mode.IMAGE,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

            self._hands = hand_landmarker.create_from_options(options)

        except Exception as exc:  # pragma: no cover - defensive, hard to trigger in tests
            raise HandTrackerError(
                f"Failed to initialize MediaPipe HandLandmarker: {exc}"
            ) from exc

        logger.info(
            "HandTracker initialized (max_hands=%s, det_conf=%s, track_conf=%s)",
            max_num_hands,
            min_detection_confidence,
            min_tracking_confidence,
        )

    def process(self, image_bgr: np.ndarray) -> TrackedHands:
        """Run detection on a single BGR frame and return TrackedHands."""
        import cv2  # local import: only needed for the color conversion here

        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb,
        )

        result = self._hands.detect(mp_image)

        hands: list[Hand] = []

        if result.hand_landmarks:
            handedness_list = result.handedness or []

            for i, hand_landmarks in enumerate(result.hand_landmarks):
                points = tuple(
                    LandmarkPoint(x=lm.x, y=lm.y, z=lm.z)
                    for lm in hand_landmarks
                )

                if len(points) != NUM_LANDMARKS:
                    logger.warning(
                        "Unexpected landmark count: %d",
                        len(points),
                    )
                    continue

                label = "Unknown"
                score = 0.0

                if i < len(handedness_list):
                    classification = handedness_list[i][0]
                    label = classification.category_name
                    score = classification.score

                hands.append(
                    Hand(
                        landmarks=points,
                        handedness=label,
                        score=score,
                    )
                )

        return TrackedHands(
            hands=tuple(hands),
            frame_width=w,
            frame_height=h,
            timestamp=time.monotonic(),
        )

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def connections(self):
        """Expose MediaPipe's hand connection topology for drawing skeletons."""
        return mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS