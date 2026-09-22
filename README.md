# AirTouch

AirTouch is a webcam-based gesture-control platform for Windows (and,
with no code changes, most other desktops). V1 tracks one hand and
turns a small set of poses and motions into mouse and keyboard input.

It is built as a **platform**, not a demo: every layer is isolated
behind an interface so that future capabilities — two-hand gestures,
machine-learning-based recognition, Blender/browser/presentation
profiles, a GUI, user-defined gestures — can be added without
rewriting the core.

```
Hand Tracking → Landmark Processing → Feature Extraction
→ Gesture Recognition → Gesture State Machine → Action Mapping
→ Action Execution
```

---

## 1. What AirTouch is

AirTouch watches your webcam, finds your hand, and maps a small set of
poses and motions to mouse/keyboard actions:

| Gesture       | Meaning                          |
|---------------|-----------------------------------|
| ☝️ INDEX_POINT | Move the cursor                   |
| 🤏 PINCH (tap) | Left click                        |
| 🤏 PINCH (hold)| Drag (press, move, release)       |
| ✊ FIST        | Right click                       |
| ✌️ TWO_FINGER  | Scroll (vertical hand movement)   |
| ✋ OPEN_HAND   | Pause AirTouch                    |
| ← Swipe left   | Previous / Back                   |
| → Swipe right  | Next / Forward                    |

## 2. Current capabilities (V1)

- Real-time single-hand tracking (21 landmarks) via MediaPipe
- 8 gestures: `OPEN_HAND`, `FIST`, `INDEX_POINT`, `PINCH`, `PINCH_HOLD`,
  `TWO_FINGER`, `SWIPE_LEFT`, `SWIPE_RIGHT`
- Absolute cursor control with configurable sensitivity, smoothing, and
  a jitter-suppressing deadzone
- Click / drag with proper state transitions (a held pinch never fires
  repeated clicks; a drag never leaves the mouse button stuck)
- Vertical two-finger scrolling with a movement deadzone
- Swipe detection (all four directions computed; LEFT/RIGHT wired to
  actions in V1)
- On-screen debug HUD: FPS, hand status, current gesture, mode,
  confidence, pinch distance, hand speed
- Emergency stop (`Q` or `ESC`), safe shutdown that always releases a
  held mouse button
- 44 unit tests covering features, gesture classification, state
  machine lifecycle, and smoothing/mapping math — all run without a
  webcam

## 3. Architecture

```
AirTouch/
├── app/            # Orchestrator: wires everything together, owns
│                   # interaction mode + cursor/scroll mapping
├── vision/         # Camera capture + MediaPipe hand tracking
├── gestures/       # Feature extraction, static/motion recognition,
│                   # gesture state machine
├── actions/        # OS-facing effects (mouse, keyboard), isolated
│                   # behind interfaces + a name-based registry
├── mappings/       # Gesture -> action name tables ("profiles")
├── config/         # Typed settings loaded from config.yaml
├── ui/             # OpenCV HUD rendering
├── utils/          # Geometry, smoothing, logging
└── tests/          # Synthetic-landmark tests, no camera required
```

### Layering rule

Each layer only knows about the layer directly below it, and never
about what's beneath *that*:

```
Vision            (mediapipe/cv2 — isolated to vision/hand_tracker.py
                    and vision/camera.py only)
   ↓  Hand / TrackedHands (plain dataclasses)
Gestures          (features → recognizer → state machine)
   ↓  GestureEvent (plain dataclass: gesture, lifecycle state, coords)
Mappings          (gesture+lifecycle → action name)
   ↓  action name (string)
Actions           (PointerController / KeyboardController — the only
                    modules that import pyautogui)
   ↓
Windows / (future) Blender / Browser / IoT
```

The gesture recognizer emits `GestureResult(gesture="PINCH",
confidence=0.94, ...)` — never `pyautogui.click()`. The action layer
never imports MediaPipe or knows what a "landmark" is. This is what
lets a future `MLGestureRecognizer` or `BlenderActions` module slot in
without touching the other layers.

### Why a state machine?

Gestures aren't just "is the hand doing X right now" — they have a
lifecycle. A pinch that's held for 50ms is a click; the same pinch
held for 500ms is a drag. `gestures/state_machine.py` turns noisy,
per-frame classifications into a clean `STARTED → ACTIVE → RELEASED`
(or `CANCELLED`) event stream, with debouncing so a single flickered
frame doesn't cancel an in-progress gesture. The pinch lifecycle
specifically is:

```
OPEN → PINCH (STARTED) → released quickly → PINCH (RELEASED) = CLICK

OPEN → PINCH (STARTED) → held past threshold → PINCH_HOLD (STARTED)
     → PINCH_HOLD (ACTIVE, moving) → released → PINCH_HOLD (RELEASED) = DRAG END
```

A pinch that becomes a drag is never also reported as a click.

## 4. Installation

Requires **Python 3.11+**.

```bash
git clone <this-repo>
cd AirTouch
python -m venv .venv
```

### Activate the virtual environment

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on MediaPipe:** MediaPipe's published wheels typically lag
> behind the newest CPython releases. `requirements.txt` pins a recent
> stable `0.10.x` release. If installation fails on your Python
> version, check MediaPipe's PyPI page for the latest version that
> supports your interpreter.

## 5. Running the application

```bash
python main.py
```

Optional custom config path:

```bash
python main.py --config path/to/config.yaml
```

A webcam window titled **AirTouch** opens showing your camera feed,
hand skeleton, and HUD. Move your hand in front of the camera to try
the gestures listed above.

**Press `Q` or `ESC` at any time to safely stop AirTouch.** Closing the
window also stops it safely.

## 6. Supported gestures (detail)

- **OPEN_HAND** — most/all fingers extended. Pauses interaction while
  held (no actions fire, HUD shows `PAUSED`).
- **FIST** — no fingers extended. Fires a single right-click when
  confirmed (not repeated while held).
- **INDEX_POINT** — only the index finger extended. Fingertip position
  drives the cursor.
- **PINCH** — thumb and index fingertip closer than the configured
  threshold. Released quickly → left click.
- **PINCH_HOLD** — a pinch held past `pinch.hold_time`. Cursor follows
  the index finger while held; releasing ends the drag.
- **TWO_FINGER** — index + middle extended, ring/pinky folded.
  Vertical hand movement scrolls.
- **SWIPE_LEFT / SWIPE_RIGHT** — a fast horizontal hand movement.
  Mapped to Back/Forward (arrow keys by default). UP/DOWN are detected
  by the same detector but unmapped in V1.

## 7. Configuration

All tunables live in `config.yaml` — nothing is hardcoded in the
gesture or action code.

```yaml
cursor:
  smoothing: 0.65    # 0..1, higher = smoother/slower
  sensitivity: 1.0    # >1 = smaller movements cover more screen
  deadzone: 0.02       # ignore jitter below this normalized distance

pinch:
  threshold: 0.05      # normalized thumb-index distance to count as pinched
  hold_time: 0.25       # seconds before a pinch becomes a drag

scroll:
  sensitivity: 2.0
  deadzone: 0.02

swipe:
  min_distance: 0.15
  max_duration: 0.6
  min_velocity: 0.4
  cooldown: 0.5
```

See `config.yaml` for the full set (camera, tracking confidence,
gesture debounce, debug, safety).

## 8. Safety controls

- **Emergency stop:** `Q` or `ESC` immediately begins shutdown.
- **Stuck-button prevention:** the mouse button is force-released on
  shutdown, on tracking loss, and whenever the gesture state machine
  reports a drag ending — even if that happens because the hand left
  the frame rather than a clean pinch release.
- **Interaction modes:** `ACTIVE`, `PAUSED` (via `OPEN_HAND`), and
  `STOPPING` are tracked explicitly in `app/state.py`. While paused,
  only the drag-release safety action still fires.
- Camera, MediaPipe, and mouse-controller failures are caught and
  reported with a clear message rather than crashing silently.

## 9. Debug mode

Enabled by default (`debug.enabled: true` in `config.yaml`). Shows, in
the HUD:

```
Hand: DETECTED
Gesture: PINCH_HOLD (ACTIVE)
Mode: ACTIVE
FPS: 29.7
Confidence: 0.94
Pinch dist: 0.041
Speed: 0.18
```

Set `debug.enabled: false` to hide the extra debug lines, or
`debug.show_landmarks: false` to hide the landmark dots (the skeleton
still draws).

## 10. Testing

No webcam required — the recognizer accepts pre-built feature data,
and `tests/helpers.py` builds synthetic 21-landmark hands for each
pose.

```bash
pip install pytest
pytest tests/ -v
```

44 tests across:

- `test_features.py` — extension detection, distances, velocity
- `test_gestures.py` — static pose classification + swipe detection
  and cooldown
- `test_state_machine.py` — debounce, click-vs-drag lifecycle, swipe
  passthrough
- `test_smoothing.py` — smoothing math, cursor/scroll mapping

## 11. Known limitations (V1)

- Single hand only (architecture supports more — see roadmap)
- Rule-based recognition only; no learned model yet
- No GUI/config editor — `config.yaml` is edited by hand
- No application-specific profiles yet (Blender/browser/presentation
  all use the same desktop mapping)
- Swipe UP/DOWN are detected but not mapped to an action
- Windows mouse control via PyAutoGUI is cross-platform but has only
  been exercised against the PyAutoGUI API surface, not
  platform-specific edge cases (multi-monitor DPI scaling, etc.)

## 12. Roadmap

```
                    AIRTOUCH
                       │
             ┌─────────┴─────────┐
             │                   │
        VISION ENGINE       INPUT SOURCES
             │                   │
        Hand Tracking       Webcam
        Face Tracking       Future sensors
        Depth               Future devices
             │
             ↓
       FEATURE ENGINE
             │
             ↓
      GESTURE ENGINE
             │
       ┌─────┴─────┐
       │           │
    Rule Based     ML
       │           │
       └─────┬─────┘
             ↓
       ACTION ENGINE
             │
   ┌─────────┼──────────┐
   ↓         ↓          ↓
Windows   Blender    Browser
Actions   Actions    Actions
   │         │          │
   └─────────┼──────────┘
             ↓
       CONFIGURATION
             ↓
      USER PROFILES
```

Near-term milestones, roughly in order:

1. Two-hand support (`vision.landmarks.TrackedHands.hands` is already
   a list — no restructuring needed)
2. A GUI for live config editing and gesture visualization
3. Per-application gesture profiles (Blender/browser/presentation),
   swapped in via `mappings/`
4. `MLGestureRecognizer` trained on logged `HandFeatures`, coexisting
   with the rule-based recognizer via `HybridGestureRecognizer`
5. User-defined custom gestures (record → label → add to a mapping)

## 13. How to add a new gesture

1. **If it's a static pose:** add a case to
   `gestures/static_gestures.classify_static`, returning a new
   `GestureName` (add it to the enum in `gestures/gesture_types.py`
   first).
2. **If it's motion-based:** follow the pattern in
   `gestures/motion_gestures.SwipeDetector` — a small stateful class
   fed `(timestamp, position)` samples, returning a `GestureName` when
   a pattern completes.
3. Wire it into `gestures/recognizer.RuleBasedGestureRecognizer.recognize`.
4. If it needs a lifecycle beyond simple STARTED/ACTIVE/RELEASED
   (like `PINCH_HOLD`'s escalation), extend
   `gestures/state_machine.GestureStateMachine`.
5. Add a test in `tests/test_gestures.py` using a synthetic hand from
   `tests/helpers.py` (add a new builder function if needed).

No changes to `actions/` are required until step 6.

## 14. How to add a new action

1. Add an `Action` subclass in the relevant `actions/` module (or a
   new module, e.g. `actions/media.py`), implementing `execute(self,
   context: ActionContext)`.
2. Register an instance in `app/application.Application._register_actions`.
3. Add an entry to `mappings/default_mappings.DEFAULT_MAPPINGS` (or a
   new profile mapping) keyed by `(GestureName, LifecycleState)`.

The gesture engine never needs to change — it only ever emits
`GestureEvent`s; the mapping table is the only place gesture-to-action
wiring lives.

## 15. How future integrations (e.g. Blender) will work

The action layer is deliberately generic: `Action.execute(context)`
doesn't know or care what application is focused. A future
`actions/blender.py` would define actions like `GrabObject`,
`RotateView`, etc., and a `mappings/blender_mappings.py` would redefine
`(GestureName.PINCH_HOLD, LifecycleState.STARTED)` to map to
`["GRAB_OBJECT"]` instead of `["START_DRAG", "MOVE_POINTER"]`.
Switching profiles (manually, or later based on the focused window)
means swapping which mapping table `app/application.py` consults —
nothing in `vision/`, `gestures/`, or the `Action` base class changes.

---

Built incrementally, phase by phase (camera → landmarks → static
gestures → cursor → pinch/click → scroll → drag → swipe), with tests
validating each phase before the next was added.
