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

Two control modes share this pipeline (toggle with the M key):

    MOUSE      one hand, classic pointer control (the original behavior)
    DESIGN_3D  one hand pinch-held orbits, two hands pinched zoom; the rest
               of the mouse mapping still applies (see
               mappings/design_3d_mappings.py)
"""

from __future__ import annotations

import time
from collections import deque

import cv2

from actions.base import ActionContext
from actions.keyboard import KeyPress, NullKeyboardController, PyAutoGUIKeyboardController
from actions.mouse import (
    EndDrag,
    EndOrbit,
    LeftClick,
    MovePointer,
    NullPointerController,
    OrbitMove,
    PointerController,
    PyAutoGUIPointerController,
    RightClick,
    Scroll,
    StartDrag,
    StartOrbit,
    Zoom,
)
from actions.registry import ActionRegistry
from app.state import CursorMapper, InteractionModeController, ScrollMapper
from config.settings import Settings
from gestures.features import FeatureExtractor, HandFeatures, MultiHandFeatureExtractor
from gestures.gesture_types import (
    ControlMode,
    GestureEvent,
    GestureName,
    GestureResult,
    InteractionMode,
)
from gestures.recognizer import RuleBasedGestureRecognizer
from gestures.state_machine import GestureStateMachine
from mappings.default_mappings import get_actions
from mappings.design_3d_mappings import get_actions_3d
from ui.overlay import draw_hud, draw_skeleton
from utils.logging import get_logger
from vision.camera import Camera, CameraError
from vision.hand_tracker import HandTracker, HandTrackerError

logger = get_logger(__name__)

_EMERGENCY_KEYS = {ord("q"), ord("Q"), 27}  # 27 == ESC
_MODE_TOGGLE_KEYS = {ord("m"), ord("M")}

# Actions that only ever RELEASE something. They must still run while
# AirTouch is paused, otherwise a button could stay held down.
_RELEASE_ACTIONS = {"END_DRAG", "END_ORBIT"}


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

    def __init__(
        self,
        settings: Settings,
        use_real_input_devices: bool = True,
        control_mode: ControlMode = ControlMode.MOUSE,
        design_pinch_hold_time: float = 0.15,
        dominant_hand_slot: str = "right",
    ) -> None:
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
        self.multi_extractor = MultiHandFeatureExtractor()
        self.recognizer = RuleBasedGestureRecognizer(
            pinch_threshold=settings.pinch.threshold,
            swipe_min_distance=settings.swipe.min_distance,
            swipe_max_duration=settings.swipe.max_duration,
            swipe_min_velocity=settings.swipe.min_velocity,
            swipe_cooldown=settings.swipe.cooldown,
            min_pinch_confidence=settings.gesture.min_confidence,
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

        # 3D design mode state.
        self.control_mode = control_mode
        self.dominant_hand_slot = dominant_hand_slot  # which hand orbits when both are visible
        self._mouse_pinch_hold_time = settings.pinch.hold_time
        self._design_pinch_hold_time = design_pinch_hold_time
        self._last_hand_count = 0
        self._warned_no_hands_attr = False
        self._apply_control_mode_settings()

    def _register_actions(self) -> None:
        reg = self.action_registry
        reg.register(MovePointer(self.pointer_controller))
        reg.register(LeftClick(self.pointer_controller))
        reg.register(RightClick(self.pointer_controller))
        reg.register(StartDrag(self.pointer_controller))
        reg.register(EndDrag(self.pointer_controller))
        reg.register(Scroll(self.pointer_controller))
        reg.register(StartOrbit(self.pointer_controller))
        reg.register(OrbitMove(self.pointer_controller))
        reg.register(EndOrbit(self.pointer_controller))
        reg.register(Zoom(self.pointer_controller))
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
        self._warn_if_hand_limit_too_low()

    def run(self) -> None:
        self.setup()
        try:
            self._loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.shutdown()

    # --- Control mode ------------------------------------------------------

    def toggle_control_mode(self) -> None:
        if self.control_mode == ControlMode.MOUSE:
            self.set_control_mode(ControlMode.DESIGN_3D)
        else:
            self.set_control_mode(ControlMode.MOUSE)

    def set_control_mode(self, mode: ControlMode) -> None:
        if mode == self.control_mode:
            return
        # Let go of anything held before the meaning of gestures changes.
        self._release_held_inputs()
        self.control_mode = mode
        self.feature_extractor.reset()
        self.multi_extractor.reset()
        self.recognizer.reset()
        self.state_machine.reset()
        self.cursor_mapper.reset()
        self.scroll_mapper.reset()
        self._last_hand_count = 0
        self._last_gesture_label = "NONE"
        self._apply_control_mode_settings()
        logger.info("Control mode: %s", mode.value)
        self._warn_if_hand_limit_too_low()

    def _apply_control_mode_settings(self) -> None:
        in_3d = self.control_mode == ControlMode.DESIGN_3D
        # Fast orbit/pan movement while pinching must not count as a swipe.
        self.recognizer.suppress_swipe_while_pinched = in_3d
        # Orbit should engage sooner than a mouse drag.
        self.state_machine.pinch_hold_time = (
            self._design_pinch_hold_time if in_3d else self._mouse_pinch_hold_time
        )

    def _warn_if_hand_limit_too_low(self) -> None:
        if (
            self.control_mode == ControlMode.DESIGN_3D
            and self.settings.tracking.max_num_hands < 2
        ):
            logger.warning(
                "3D mode needs two-hand tracking but max_num_hands is %d; "
                "set it to 2 in config.yaml to enable two-hand zoom",
                self.settings.tracking.max_num_hands,
            )

    # --- Frame loop --------------------------------------------------------

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
                if self.control_mode == ControlMode.DESIGN_3D:
                    result, features = self._recognize_3d(tracked)
                else:
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
                    if self.control_mode == ControlMode.DESIGN_3D:
                        debug_lines += [
                            f"Hands: {self._last_hand_count}",
                            f"Delta: ({result.delta[0]:+.4f}, {result.delta[1]:+.4f})",
                            f"Pair dist delta: {result.pair_distance_delta:+.4f}",
                        ]
            else:
                self.feature_extractor.reset()
                self.multi_extractor.reset()
                self.recognizer.reset()
                self.state_machine.reset()
                self.interaction.on_tracking_lost()
                # Never leave a mouse button stuck down if tracking is lost.
                self._release_held_inputs()
                self.cursor_mapper.reset()
                self.scroll_mapper.reset()
                self._last_hand_count = 0
                self._last_gesture_label = "NONE"

            if self.settings.debug.show_landmarks or True:
                draw_skeleton(
                    frame.image,
                    tracked,
                    self.hand_tracker.connections,
                    show_landmarks=self.settings.debug.show_landmarks,
                )
            mode_label = self.interaction.mode.value
            if self.control_mode == ControlMode.DESIGN_3D:
                mode_label = f"{mode_label} | 3D"
            draw_hud(
                frame.image,
                fps=fps,
                hand_detected=tracked.detected,
                gesture=self._last_gesture_label,
                mode=mode_label,
                debug_lines=debug_lines if self.settings.debug.enabled else None,
            )

            cv2.imshow(self.WINDOW_NAME, frame.image)
            key = cv2.waitKey(1) & 0xFF
            if key in _EMERGENCY_KEYS:
                logger.info("Emergency stop key pressed")
                self.interaction.request_stop()
            elif key in _MODE_TOGGLE_KEYS:
                self.toggle_control_mode()

            # Window-close (X button) also ends the loop safely.
            if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self.interaction.request_stop()

    def _hands_of(self, tracked) -> list:
        """All tracked hands (at most two). Falls back to the primary hand
        only if the tracker result exposes no ``hands`` collection."""
        hands = getattr(tracked, "hands", None)
        if hands:
            return list(hands)[:2]
        if not self._warned_no_hands_attr:
            logger.warning(
                "Tracker result has no 'hands' attribute; "
                "two-hand control is disabled (primary hand only)"
            )
            self._warned_no_hands_attr = True
        return [tracked.primary]

    def _recognize_3d(self, tracked) -> tuple[GestureResult, HandFeatures]:
        """3D-mode recognition: a two-hand pinch wins when both hands are
        pinched, otherwise the dominant hand is recognized on its own.
        Exactly one GestureResult per frame goes to the state machine."""
        multi = self.multi_extractor.extract(self._hands_of(tracked), timestamp=tracked.timestamp)

        # When the number of visible hands changes, the hand in each slot
        # may be a different hand. Drop motion state so nothing jumps.
        count = len(multi.hands)
        if count != self._last_hand_count:
            self.recognizer.reset()
            self._last_hand_count = count

        primary = multi.hands.get(self.dominant_hand_slot) or next(iter(multi.hands.values()))

        if multi.pair is not None:
            pair_result = self.recognizer.recognize_pair(multi.pair)
            if pair_result.gesture == GestureName.TWO_HAND_PINCH:
                return pair_result, primary

        return self.recognizer.recognize(primary), primary

    def _release_held_inputs(self) -> None:
        context = ActionContext(timestamp=time.monotonic())
        for action_name in sorted(_RELEASE_ACTIONS):
            try:
                self.action_registry.execute(action_name, context)
            except Exception:
                logger.exception("Error running %s while releasing held inputs", action_name)

    def _handle_event(self, event: GestureEvent) -> None:
        self.interaction.on_gesture_event(event)
        mode = self.interaction.mode
        lookup = get_actions_3d if self.control_mode == ControlMode.DESIGN_3D else get_actions
        for action_name in lookup(event.gesture, event.state):
            if mode != InteractionMode.ACTIVE and action_name not in _RELEASE_ACTIONS:
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
        if action_name == "ORBIT_MOVE":
            return ActionContext(
                timestamp=event.timestamp,
                extra={"dx": event.delta[0], "dy": event.delta[1]},
            )
        if action_name == "ZOOM":
            return ActionContext(
                timestamp=event.timestamp,
                extra={"zoom_delta": event.pair_distance_delta},
            )
        return ActionContext(timestamp=event.timestamp)

    def shutdown(self) -> None:
        logger.info("Shutting down AirTouch")
        # Always release held mouse buttons, even if a drag or orbit was in progress.
        self._release_held_inputs()
        if self.hand_tracker is not None:
            self.hand_tracker.close()
        self.camera.close()
        cv2.destroyAllWindows()