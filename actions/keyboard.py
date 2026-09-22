"""Keyboard control, isolated behind ``KeyboardController`` the same way
mouse control is isolated behind ``PointerController``.

V1 only uses this for swipe-triggered navigation keys (e.g. Back/
Forward), but it's general enough for future media-key and shortcut
actions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from actions.base import Action, ActionContext
from utils.logging import get_logger

logger = get_logger(__name__)


class KeyboardController(ABC):
    @abstractmethod
    def press_key(self, key: str) -> None:
        ...


class PyAutoGUIKeyboardController(KeyboardController):
    def __init__(self) -> None:
        import pyautogui

        self._pyautogui = pyautogui

    def press_key(self, key: str) -> None:
        try:
            self._pyautogui.press(key)
        except Exception:
            logger.exception("Key press failed for key=%s", key)


class NullKeyboardController(KeyboardController):
    """No-op controller for tests and headless environments."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def press_key(self, key: str) -> None:
        self.calls.append(key)


class KeyPress(Action):
    name = "KEY_PRESS"

    def __init__(self, controller: KeyboardController, key: str) -> None:
        self._controller = controller
        self._key = key

    def execute(self, context: ActionContext) -> None:
        self._controller.press_key(self._key)
