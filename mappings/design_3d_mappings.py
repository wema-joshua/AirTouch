"""Mappings for 3D design mode.

3D mode is the normal mouse mapping with a handful of overrides on top:
anything not listed here (pointing to move the cursor, quick pinch to
click, two-finger scroll, ...) falls through to ``get_actions`` from
``mappings.default_mappings`` unchanged. Only the gestures whose meaning
differs in 3D are redefined.
"""

from __future__ import annotations

from collections.abc import Iterable

from gestures.gesture_types import GestureName, LifecycleState
from mappings.default_mappings import get_actions

_G = GestureName
_S = LifecycleState

_OVERRIDES: dict[tuple[GestureName, LifecycleState], list[str]] = {
    # One hand, pinch held: orbit the view (middle-mouse drag).
    (_G.PINCH_HOLD, _S.STARTED): ["START_ORBIT"],
    (_G.PINCH_HOLD, _S.ACTIVE): ["ORBIT_MOVE"],
    (_G.PINCH_HOLD, _S.RELEASED): ["END_ORBIT"],
    (_G.PINCH_HOLD, _S.CANCELLED): ["END_ORBIT"],
    # Both hands pinched: hands apart / together zooms.
    (_G.TWO_HAND_PINCH, _S.ACTIVE): ["ZOOM"],
    # Swipes would send Left/Right arrow keys into the 3D app.
    (_G.SWIPE_LEFT, _S.ACTIVE): [],
    (_G.SWIPE_RIGHT, _S.ACTIVE): [],
    (_G.SWIPE_UP, _S.ACTIVE): [],
    (_G.SWIPE_DOWN, _S.ACTIVE): [],
}


def get_actions_3d(gesture: GestureName, state: LifecycleState) -> Iterable[str]:
    """Action names for a gesture event while in 3D design mode."""
    override = _OVERRIDES.get((gesture, state))
    if override is not None:
        return override
    return get_actions(gesture, state)