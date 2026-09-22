"""Gesture-to-action mapping tables.

A "profile" is just a dict of (gesture, lifecycle state) -> action
names. Swapping profiles (Desktop / Blender / Browser / Presentation)
is how future application-specific gesture mappings work — the gesture
engine and action registry never change, only which mapping table is
active.
"""
