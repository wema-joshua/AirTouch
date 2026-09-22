"""AirTouch application orchestrator.

This is the only module that knows about every layer (vision, gestures,
actions, mappings, UI). It is intentionally thin: it wires components
together and drives the frame loop, but contains no gesture-recognition
or action-execution logic of its own — that all lives in the layers it
composes.

Pipeline per frame:

    Camera -> HandTracker -> FeatureExtractor -> GestureRecognizer
    -> GestureStateMachine -> (InteractionModeController + ActionRegistry)
    -> Overlay
"""

from __future__ import annotations

import time
from collections import deque

import cv2

from actions.base import ActionContext
from actions.keyboard import KeyPress, NullKeyboardController, PyAutoGUIKeyboardController
from actions.mouse import (
    EndDrag,
    LeftClick,
    MovePointer,
    NullPointerController,
    PointerController,
    PyAutoGUIPointerController,
    RightClick,
    Scroll,
    StartDrag,
)
from actions.registry import ActionRegistry
from app.state import CursorMapper, InteractionModeController, ScrollMapper
from config.settings import Settings
from gestures.features import FeatureExtractor
from gestures.gesture_types import GestureEvent, InteractionMode
from gestures.recognizer import RuleBasedGestureRecognizer
from gestures.state_machine import GestureStateMachine
from mappings.default_mappings import get_actions
from ui.overlay import draw_hud, draw_skeleton
from utils.logging import get_logger
from vision.camera import Camera, CameraError
from vision.hand_tracker import HandTracker, HandTrackerError

logger = get_logger(__name__)

_EMERGENCY_KEYS = {ord("q"), ord("Q"), 27}  # 27 == ESC


class _FpsCounter:
    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        now = time.monotonic()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / span


class Application:
    """Owns the full lifecycle: setup, run loop, and teardown."""

    WINDOW_NAME = "AirTouch"

    def __init__(self, settings: Settings, use_real_input_devices: bool = True) -> None:
        self.settings = settings
        self._use_real_input_devices = use_real_input_devices

        self.camera = Camera(
            device=settings.camera.device,
            width=settings.camera.width,
            height=settings.camera.height,
            fps=settings.camera.fps,
            mirrored=settings.camera.mirrored,
        )
        self.hand_tracker: HandTracker | None = None
        self.feature_extractor = FeatureExtractor()
        self.recognizer = RuleBasedGestureRecognizer(
            pinch_threshold=settings.pinch.threshold,
            swipe_min_distance=settings.swipe.min_distance,
            swipe_max_duration=settings.swipe.max_duration,
            swipe_min_velocity=settings.swipe.min_velocity,
            swipe_cooldown=settings.swipe.cooldown,
        )
        self.state_machine = GestureStateMachine(
            min_confidence=settings.gesture.min_confidence,
            hold_frames=settings.gesture.static_hold_frames,
            pinch_hold_time=settings.pinch.hold_time,
        )
        self.interaction = InteractionModeController()
        self.action_registry = ActionRegistry()

        self.pointer_controller: PointerController
        if use_real_input_devices:
            self.pointer_controller = PyAutoGUIPointerController()
            self.keyboard_controller = PyAutoGUIKeyboardController()
        else:
            self.pointer_controller = NullPointerController()
            self.keyboard_controller = NullKeyboardController()

        screen_w, screen_h = self.pointer_controller.screen_size()
        self.cursor_mapper = CursorMapper(
            screen_width=screen_w,
            screen_height=screen_h,
            sensitivity=settings.cursor.sensitivity,
            deadzone=settings.cursor.deadzone,
            smoothing=settings.cursor.smoothing,
        )
        self.scroll_mapper = ScrollMapper(
            sensitivity=settings.scroll.sensitivity,
            deadzone=settings.scroll.deadzone,
            invert=settings.scroll.invert,
        )

        self._register_actions()
        self._fps_counter = _FpsCounter()
        self._last_gesture_label = "NONE"

    def _register_actions(self) -> None:
        reg = self.action_registry
        reg.register(MovePointer(self.pointer_controller))
        reg.register(LeftClick(self.pointer_controller))
        reg.register(RightClick(self.pointer_controller))
        reg.register(StartDrag(self.pointer_controller))
        reg.register(EndDrag(self.pointer_controller))
        reg.register(Scroll(self.pointer_controller))
        # "PREVIOUS"/"NEXT" are named actions (not "KEY_PRESS") so the
        # mapping table can address them directly; swap the bound key
        # here to retarget swipes without touching gesture code.
        prev_action = KeyPress(self.keyboard_controller, "left")
        prev_action.name = "PREVIOUS"
        next_action = KeyPress(self.keyboard_controller, "right")
        next_action.name = "NEXT"
        reg.register(prev_action)
        reg.register(next_action)

    def setup(self) -> None:
        try:
            self.camera.open()
        except CameraError as exc:
            logger.error(str(exc))
            raise
        try:
            self.hand_tracker = HandTracker(
                min_detection_confidence=self.settings.tracking.min_detection_confidence,
                min_tracking_confidence=self.settings.tracking.min_tracking_confidence,
                max_num_hands=self.settings.tracking.max_num_hands,
            )
        except HandTrackerError as exc:
            logger.error(str(exc))
            self.camera.close()
            raise

    def run(self) -> None:
        self.setup()
        try:
            self._loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.shutdown()

    def _loop(self) -> None:
        assert self.hand_tracker is not None
        while self.interaction.mode != InteractionMode.STOPPING:
            frame = self.camera.read()
            if frame is None:
                logger.warning("Failed to read frame from camera; retrying")
                continue

            tracked = self.hand_tracker.process(frame.image)
            fps = self._fps_counter.tick()

            debug_lines: list[str] = []
            if tracked.detected:
                hand = tracked.primary
                features = self.feature_extractor.extract(hand, timestamp=tracked.timestamp)
                result = self.recognizer.recognize(features)
                events = self.state_machine.update(result)
                for event in events:
                    self._handle_event(event)
                    self._last_gesture_label = f"{event.gesture.value} ({event.state.value})"

                if self.settings.debug.enabled:
                    debug_lines = [
                        f"Confidence: {result.confidence:.2f}",
                        f"Pinch dist: {result.thumb_index_distance:.3f}",
                        f"Speed: {features.speed:.3f}",
                    ]
            else:
                self.feature_extractor.reset()
                self.recognizer.reset()
                self.state_machine.reset()
                self.interaction.on_tracking_lost()
                # Never leave the mouse button stuck down if tracking is lost.
                self.action_registry.execute("END_DRAG", ActionContext(timestamp=time.monotonic()))
                self.cursor_mapper.reset()
                self.scroll_mapper.reset()
                self._last_gesture_label = "NONE"

            if self.settings.debug.show_landmarks or True:
                draw_skeleton(
                    frame.image,
                    tracked,
                    self.hand_tracker.connections,
                    show_landmarks=self.settings.debug.show_landmarks,
                )
            draw_hud(
                frame.image,
                fps=fps,
                hand_detected=tracked.detected,
                gesture=self._last_gesture_label,
                mode=self.interaction.mode.value,
                debug_lines=debug_lines if self.settings.debug.enabled else None,
            )

            cv2.imshow(self.WINDOW_NAME, frame.image)
            key = cv2.waitKey(1) & 0xFF
            if key in _EMERGENCY_KEYS:
                logger.info("Emergency stop key pressed")
                self.interaction.request_stop()

            # Window-close (X button) also ends the loop safely.
            if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self.interaction.request_stop()

    def _handle_event(self, event: GestureEvent) -> None:
        self.interaction.on_gesture_event(event)
        mode = self.interaction.mode
        for action_name in get_actions(event.gesture, event.state):
            if mode != InteractionMode.ACTIVE and action_name != "END_DRAG":
                continue
            context = self._build_context(action_name, event)
            self.action_registry.execute(action_name, context)

    def _build_context(self, action_name: str, event: GestureEvent) -> ActionContext:
        if action_name == "MOVE_POINTER":
            screen_x, screen_y = self.cursor_mapper.update(event.index_tip)
            return ActionContext(
                timestamp=event.timestamp,
                screen_x=screen_x,
                screen_y=screen_y,
                norm_x=event.index_tip[0],
                norm_y=event.index_tip[1],
                velocity_x=event.velocity[0],
                velocity_y=event.velocity[1],
            )
        if action_name == "SCROLL":
            amount = self.scroll_mapper.update(event.velocity[1])
            return ActionContext(timestamp=event.timestamp, extra={"scroll_amount": amount})
        return ActionContext(timestamp=event.timestamp)

    def shutdown(self) -> None:
        logger.info("Shutting down AirTouch")
        # Always release the mouse button, even if a drag was in progress.
        try:
            self.action_registry.execute("END_DRAG", ActionContext(timestamp=time.monotonic()))
        except Exception:
            logger.exception("Error releasing mouse button during shutdown")
        if self.hand_tracker is not None:
            self.hand_tracker.close()
        self.camera.close()
        cv2.destroyAllWindows()
