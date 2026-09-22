"""Generic action abstraction.

The gesture engine never calls the OS directly. Instead, gesture events
are mapped (see ``mappings/``) to ``Action`` instances, which are given
an ``ActionContext`` and asked to ``execute``. This is the seam where
future ``BlenderActions``, ``BrowserActions``, etc. plug in: the same
gesture ("PINCH + MOVE") can be mapped to different Action
implementations depending on the active application profile, without
the gesture engine changing at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionContext:
    """Everything an Action might need, pre-computed by the app layer.

    Fields are optional/defaulted because different actions need
    different subsets (a KeyPress doesn't need screen coordinates, a
    Scroll doesn't need a key name, etc.).
    """

    timestamp: float = 0.0

    # Screen-space (pixels), already mapped from normalized hand
    # coordinates — actions never do that mapping themselves.
    screen_x: float = 0.0
    screen_y: float = 0.0

    # Normalized (0..1) hand-space position/velocity, for actions that
    # want relative rather than absolute values (e.g. scroll delta).
    norm_x: float = 0.0
    norm_y: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    key: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Action(ABC):
    """A single, named, executable effect."""

    name: str = "UNNAMED_ACTION"

    @abstractmethod
    def execute(self, context: ActionContext) -> None:
        """Perform the action's effect. Must not raise for expected,
        recoverable conditions — log and return instead."""
