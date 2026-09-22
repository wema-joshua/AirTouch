"""Default desktop gesture -> action mapping for V1.

Keyed by (GestureName, LifecycleState) so the same gesture can trigger
different actions depending on its lifecycle stage (e.g. PINCH only
fires a click on RELEASED, never repeatedly while held).

This is exactly the seam described in the roadmap for application
profiles: a future ``BLENDER_MAPPINGS`` table could redefine
``(PINCH_HOLD, LifecycleState.STARTED): ["GRAB_OBJECT"]`` without the
gesture engine or action registry changing at all.
"""

from __future__ import annotations

from gestures.gesture_types import GestureName, LifecycleState

GestureKey = tuple[GestureName, LifecycleState]

DEFAULT_MAPPINGS: dict[GestureKey, list[str]] = {
    # Pointer movement while pointing.
    (GestureName.INDEX_POINT, LifecycleState.STARTED): ["MOVE_POINTER"],
    (GestureName.INDEX_POINT, LifecycleState.ACTIVE): ["MOVE_POINTER"],
    # A pinch that releases before becoming a hold is a click.
    (GestureName.PINCH, LifecycleState.RELEASED): ["LEFT_CLICK"],
    # A pinch held past the hold-time threshold becomes a drag.
    (GestureName.PINCH_HOLD, LifecycleState.STARTED): ["START_DRAG", "MOVE_POINTER"],
    (GestureName.PINCH_HOLD, LifecycleState.ACTIVE): ["MOVE_POINTER"],
    (GestureName.PINCH_HOLD, LifecycleState.RELEASED): ["END_DRAG"],
    # Fist triggers a right click once when the gesture is confirmed,
    # not on every frame it's held.
    (GestureName.FIST, LifecycleState.STARTED): ["RIGHT_CLICK"],
    # Two-finger vertical movement scrolls.
    (GestureName.TWO_FINGER, LifecycleState.STARTED): ["SCROLL"],
    (GestureName.TWO_FINGER, LifecycleState.ACTIVE): ["SCROLL"],
    # Swipes are discrete navigation events.
    (GestureName.SWIPE_LEFT, LifecycleState.ACTIVE): ["PREVIOUS"],
    (GestureName.SWIPE_RIGHT, LifecycleState.ACTIVE): ["NEXT"],
    # OPEN_HAND deliberately has no action-layer mapping: it's handled
    # at the application level as a pause signal, not routed through
    # the action registry.
}


def get_actions(gesture: GestureName, state: LifecycleState) -> list[str]:
    return DEFAULT_MAPPINGS.get((gesture, state), [])
