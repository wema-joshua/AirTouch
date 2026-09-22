"""Mouse/pointer control, isolated behind ``PointerController``.

The gesture and action-mapping layers never import pyautogui directly —
only this module does. Swapping to a different automation library, or
adding a Blender/browser-specific pointer backend later, means editing
only this file (and registering the new implementation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from actions.base import Action, ActionContext
from utils.logging import get_logger

logger = get_logger(__name__)


class PointerController(ABC):
    """Interface for anything that can move a cursor and click."""

    @abstractmethod
    def move(self, x: float, y: float) -> None:
        ...

    @abstractmethod
    def click(self) -> None:
        ...

    @abstractmethod
    def right_click(self) -> None:
        ...

    @abstractmethod
    def press(self) -> None:
        """Press and hold the left button (start of a drag)."""

    @abstractmethod
    def release(self) -> None:
        """Release the left button (end of a drag)."""

    @abstractmethod
    def scroll(self, amount: int) -> None:
        """Scroll vertically. Positive = up, negative = down (OS convention)."""

    @abstractmethod
    def screen_size(self) -> tuple[int, int]:
        ...


class PyAutoGUIPointerController(PointerController):
    """Windows/cross-platform pointer control via PyAutoGUI."""

    def __init__(self) -> None:
        import pyautogui

        pyautogui.FAILSAFE = False  # AirTouch has its own emergency stop (Q/ESC)
        pyautogui.PAUSE = 0
        self._pyautogui = pyautogui
        self._dragging = False

    def move(self, x: float, y: float) -> None:
        try:
            self._pyautogui.moveTo(x, y, _pause=False)
        except Exception:
            logger.exception("Pointer move failed")

    def click(self) -> None:
        try:
            self._pyautogui.click()
        except Exception:
            logger.exception("Left click failed")

    def right_click(self) -> None:
        try:
            self._pyautogui.rightClick()
        except Exception:
            logger.exception("Right click failed")

    def press(self) -> None:
        try:
            self._pyautogui.mouseDown()
            self._dragging = True
        except Exception:
            logger.exception("Mouse down failed")

    def release(self) -> None:
        try:
            if self._dragging:
                self._pyautogui.mouseUp()
                self._dragging = False
        except Exception:
            logger.exception("Mouse up failed")

    def scroll(self, amount: int) -> None:
        try:
            self._pyautogui.scroll(amount)
        except Exception:
            logger.exception("Scroll failed")

    def screen_size(self) -> tuple[int, int]:
        size = self._pyautogui.size()
        return (size.width, size.height)

    @property
    def is_dragging(self) -> bool:
        return self._dragging


class NullPointerController(PointerController):
    """No-op controller for tests and headless environments."""

    def __init__(self, screen_w: int = 1920, screen_h: int = 1080) -> None:
        self._screen_w = screen_w
        self._screen_h = screen_h
        self.calls: list[tuple[str, tuple]] = []

    def move(self, x: float, y: float) -> None:
        self.calls.append(("move", (x, y)))

    def click(self) -> None:
        self.calls.append(("click", ()))

    def right_click(self) -> None:
        self.calls.append(("right_click", ()))

    def press(self) -> None:
        self.calls.append(("press", ()))

    def release(self) -> None:
        self.calls.append(("release", ()))

    def scroll(self, amount: int) -> None:
        self.calls.append(("scroll", (amount,)))

    def screen_size(self) -> tuple[int, int]:
        return (self._screen_w, self._screen_h)


# --- Actions -----------------------------------------------------------


class MovePointer(Action):
    name = "MOVE_POINTER"

    def __init__(self, controller: PointerController) -> None:
        self._controller = controller

    def execute(self, context: ActionContext) -> None:
        self._controller.move(context.screen_x, context.screen_y)


class LeftClick(Action):
    name = "LEFT_CLICK"

    def __init__(self, controller: PointerController) -> None:
        self._controller = controller

    def execute(self, context: ActionContext) -> None:
        self._controller.click()


class RightClick(Action):
    name = "RIGHT_CLICK"

    def __init__(self, controller: PointerController) -> None:
        self._controller = controller

    def execute(self, context: ActionContext) -> None:
        self._controller.right_click()


class StartDrag(Action):
    name = "START_DRAG"

    def __init__(self, controller: PointerController) -> None:
        self._controller = controller

    def execute(self, context: ActionContext) -> None:
        self._controller.press()


class EndDrag(Action):
    name = "END_DRAG"

    def __init__(self, controller: PointerController) -> None:
        self._controller = controller

    def execute(self, context: ActionContext) -> None:
        self._controller.release()


class Scroll(Action):
    name = "SCROLL"

    def __init__(self, controller: PointerController) -> None:
        self._controller = controller

    def execute(self, context: ActionContext) -> None:
        amount = int(context.extra.get("scroll_amount", 0))
        if amount != 0:
            self._controller.scroll(amount)
