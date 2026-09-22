"""Gesture engine: feature extraction, recognition, and state tracking.

Pipeline: Hand -> HandFeatures -> GestureResult -> GestureEvent (via the
state machine). This layer must never import mediapipe, cv2, pyautogui,
or any OS/action library — it only consumes ``vision.landmarks`` types
and produces its own plain data structures.
"""
