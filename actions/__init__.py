"""Action layer: OS-facing effects, decoupled from gestures and vision.

Nothing in this package may import mediapipe or vision.landmarks. It
only knows about ActionContext and its own Action/registry types.
"""
