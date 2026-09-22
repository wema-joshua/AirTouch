"""Action registry.

Gesture-to-action mappings (see ``mappings/default_mappings.py``) refer
to actions by name (a plain string), never by importing a concrete
Action class directly. This registry is the indirection point: it's
what lets a future "Blender mode" swap ``START_DRAG`` for a
``GrabObject`` action without the mapping table's structure changing.
"""

from __future__ import annotations

from actions.base import Action, ActionContext
from utils.logging import get_logger

logger = get_logger(__name__)


class ActionRegistry:
    def __init__(self) -> None:
        self._actions: dict[str, Action] = {}

    def register(self, action: Action) -> None:
        self._actions[action.name] = action

    def has(self, name: str) -> bool:
        return name in self._actions

    def execute(self, name: str, context: ActionContext) -> None:
        action = self._actions.get(name)
        if action is None:
            logger.warning("No action registered for name=%s", name)
            return
        try:
            action.execute(context)
        except Exception:
            logger.exception("Action %s raised during execute()", name)

    def names(self) -> list[str]:
        return list(self._actions.keys())
