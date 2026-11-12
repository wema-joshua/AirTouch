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

    @abstractmethod
    def press_button(self, button: str) -> None:
        """Press and hold 'left' | 'middle' | 'right'."""

    @abstractmethod
    def release_button(self, button: str) -> None:
        """Release 'left' | 'middle' | 'right'."""

    @abstractmethod
    def move_relative(self, dx: int, dy: int) -> None:
        """Move the cursor by a pixel delta (used for orbit/pan)."""


class PyAutoGUIPointerController(PointerController):
    """Windows/cross-platform pointer control via PyAutoGUI."""

    def __init__(self) -> None:
        import pyautogui

        pyautogui.FAILSAFE = False  # AirTouch has its own emergency stop (Q/ESC)
        pyautogui.PAUSE = 0
        self._pyautogui = pyautogui
        self._dragging = False
        self._held: set[str] = set()

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

    def press_button(self, button: str) -> None:
        try:
            self._pyautogui.mouseDown(button=button)
            self._held.add(button)
        except Exception:
            logger.exception("Mouse down (%s) failed", button)

    def release_button(self, button: str) -> None:
        try:
            if button in self._held:
                self._pyautogui.mouseUp(button=button)
                self._held.discard(button)
        except Exception:
            logger.exception("Mouse up (%s) failed", button)

    def move_relative(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        try:
            self._pyautogui.moveRel(dx, dy, _pause=False)
        except Exception:
            logger.exception("Relative move failed")

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

    def press_button(self, button: str) -> None:
        self.calls.append(("press_button", (button,)))

    def release_button(self, button: str) -> None:
        self.calls.append(("release_button", (button,)))

    def move_relative(self, dx: int, dy: int) -> None:
        self.calls.append(("move_relative", (dx, dy)))


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


# --- 3D actions --------------------------------------------------------


class StartOrbit(Action):
    name = "START_ORBIT"

    def __init__(self, controller: PointerController) -> None:
        self._controller = controller

    def execute(self, context: ActionContext) -> None:
        self._controller.press_button("middle")


class OrbitMove(Action):
    name = "ORBIT_MOVE"
    SENSITIVITY = 1200  # pixels per full-frame hand travel; tune this

    def __init__(self, controller: PointerController) -> None:
        self._controller = controller

    def execute(self, context: ActionContext) -> None:
        dx = context.extra.get("dx", 0.0) * self.SENSITIVITY
        dy = context.extra.get("dy", 0.0) * self.SENSITIVITY
        self._controller.move_relative(int(dx), int(dy))


class EndOrbit(Action):
    name = "END_ORBIT"

    def __init__(self, controller: PointerController) -> None:
        self._controller = controller

    def execute(self, context: ActionContext) -> None:
        self._controller.release_button("middle")


class Zoom(Action):
    name = "ZOOM"
    SENSITIVITY = 6.0  # scroll notches per unit of normalized hand-distance change; tune this

    def __init__(self, controller: PointerController) -> None:
        self._controller = controller
        self._remainder = 0.0

    def execute(self, context: ActionContext) -> None:
        # Hands apart (positive delta) scrolls up, which zooms in for Blender.
        # Fractional notches accumulate so slow movement still zooms.
        self._remainder += context.extra.get("zoom_delta", 0.0) * self.SENSITIVITY
        notches = int(self._remainder)
        if notches != 0:
            self._remainder -= notches
            self._controller.scroll(notches)