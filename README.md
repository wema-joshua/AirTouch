# AirTouch

Control your computer with hand gestures, using only a webcam.

AirTouch tracks your hand with MediaPipe and turns gestures into real mouse and keyboard input. It started as a touchless mouse (move, click, drag, scroll, swipe) and now also works as a **3D design controller**: orbit and zoom the viewport in Blender and other 3D software with one or two hands.

- Pure Python, runs locally, no cloud services
- Layered design: vision, gestures and actions never import each other's internals
- Two control modes (mouse and 3D design), switchable at runtime
- Built to be extended: new gestures, new actions and even a machine-learning recognizer plug in without rewrites

Developed and tested on Windows.

---

## Quick start

### 1. Install

```powershell
git clone <your-repo-url>
cd AirTouch
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

### 2. Download the hand model

The MediaPipe model file is not bundled with the pip package. Download it once into `models/`:

```powershell
mkdir models -Force
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "models\hand_landmarker.task"
```

On macOS or Linux: `curl -L -o models/hand_landmarker.task <same URL>`.

The file should be roughly 7 to 8 MB.

### 3. Run

Run from the project root, because the model path is relative:

```powershell
python main.py
```

### Controls

| Key | Action |
|---|---|
| `Q` or `Esc` | Emergency stop (also closing the window) |
| `M` | Switch between mouse mode and 3D design mode |

---

## Gestures

| Gesture | How it looks |
|---|---|
| `PINCH` | Thumb and index fingertips touching |
| `PINCH_HOLD` | A pinch held longer than `pinch.hold_time` |
| `TWO_HAND_PINCH` | Both hands pinched at the same time |
| `INDEX_POINT` | Index finger extended, others curled |
| `TWO_FINGER` | Index and middle fingers extended |
| `OPEN_HAND` | Four or five fingers extended |
| `FIST` | No fingers extended |
| `SWIPE_LEFT/RIGHT/UP/DOWN` | A fast hand movement in one direction |

### Mouse mode (default)

- A quick pinch is a left click.
- A pinch held is a drag (press on hold, release when you let go).
- Swipe left / right press the Left / Right arrow keys (handy for slides).
- Pointing, scrolling and pausing are configured in `mappings/default_mappings.py`, which is the source of truth for every gesture-to-action binding.

### 3D design mode

Press `M` to enter. This mode is the mouse mapping with a few overrides (`mappings/design_3d_mappings.py`), so pointing and click-to-select still work.

| Input | Result |
|---|---|
| One hand, pinch and hold, then move | **Orbit** the viewport (middle-mouse drag) |
| Both hands pinched, move apart / together | **Zoom** in / out (mouse wheel) |
| Quick pinch tap | Click / select |
| Swipes | Disabled, so they don't send arrow keys to your 3D app |

Notes:

- Set `tracking.max_num_hands` to `2` in `config.yaml` for two-hand zoom. A warning is logged in 3D mode if it isn't.
- With two hands visible, the dominant hand (the right side of the image by default) is the one that orbits.
- The 3D app window must have focus, because input is injected at OS level. If the app runs as administrator, run AirTouch as administrator too.
- Moving the cursor while orbiting is expected: orbiting is a middle-mouse drag.

---

## How it works

Every camera frame flows through the same pipeline:

```
Camera -> HandTracker -> FeatureExtractor -> GestureRecognizer
       -> GestureStateMachine -> ActionRegistry -> OS input
                                      ^
                              mappings (gesture + state -> action name)
```

| Stage | Responsibility |
|---|---|
| `vision/` | Opens the camera and runs MediaPipe `HandLandmarker`, returning typed `Hand` objects with 21 landmarks each |
| `gestures/features.py` | Turns landmarks into normalized numbers: pinch distance, finger extension, palm center, velocity. For two hands it also computes the distance, midpoint and angle between them |
| `gestures/static_gestures.py` | Classifies one frame's pose (pinch, fist, point, ...). Pure functions, no memory of earlier frames |
| `gestures/motion_gestures.py` | Swipe detection, plus smoothed frame-to-frame delta trackers used for orbit and zoom |
| `gestures/recognizer.py` | Combines the above into one `GestureResult` per frame. This is the seam where a future ML recognizer plugs in |
| `gestures/state_machine.py` | Debounces results and adds a lifecycle: `STARTED`, `ACTIVE`, `RELEASED`, `CANCELLED` |
| `mappings/` | Pure lookup tables: (gesture, lifecycle state) to a list of action names |
| `actions/` | The only code that touches the OS (mouse, keyboard). Each action is a small class registered by name |
| `app/application.py` | The only module that knows every layer. It wires them together and runs the frame loop |
| `ui/overlay.py` | Draws the skeleton and the on-screen status display |

### Design principles

1. **Layers don't reach across each other.** Gesture code never imports `pyautogui`. Action code never reads landmarks.
2. **Features are normalized.** Distances are divided by the hand's own size, so thresholds work at any distance from the camera.
3. **One result per frame goes to the state machine.** In 3D mode the two-hand result wins when both hands are pinched, otherwise the dominant hand is recognized alone.
4. **Everything that can hold a button also releases it.** Mode switches, tracking loss, shutdown and pausing all release held buttons, so a crash mid-orbit can't leave the mouse stuck.
5. **OS input sits behind `PointerController` / keyboard controllers**, with `Null*` versions for tests and headless runs.

### The pinch lifecycle

```
OPEN -> PINCH (STARTED) -> hold_time passes -> PINCH_HOLD (STARTED)
     -> PINCH_HOLD (ACTIVE, every frame)    -> released -> PINCH_HOLD (RELEASED)

OPEN -> PINCH (STARTED) -> released early   -> PINCH (RELEASED)  = a click
```

In mouse mode `PINCH_HOLD` is a drag. In 3D mode it is an orbit. Same gesture, different mapping table.

### Smoothing and deltas

Raw landmark positions jitter, which is barely visible when moving a cursor but very visible when rotating a 3D model.

- `utils/smoothing.py` implements the **One Euro filter**: it removes jitter when the hand is still and stays responsive when it moves fast.
- `DeltaTracker` and `ScalarDeltaTracker` (in `gestures/motion_gestures.py`) smooth a position and return the **change since the last frame**. They return zero on the first frame after a reset, so nothing jumps when you re-engage a gesture.
- An optional deadzone withholds tiny movements until they add up, so slow deliberate movement still gets through.

### Pinch hysteresis

A pinch engages at one distance and only releases at a wider one (`pinch_release_ratio`, default 1.3), so it doesn't flicker at the boundary. The state machine ignores any gesture below `gesture.min_confidence`, and pinch confidence is `1 - distance / pinch_threshold`. In practice a pinch registers once the distance is under `pinch.threshold * (1 - min_confidence)`. When tuning `pinch.threshold`, log `thumb_index_distance` while pinching and while relaxed, and set it from those readings.

---

## Project layout

```
AirTouch/
├── main.py                    Entry point
├── config.yaml                All tunable settings
├── requirements.txt
├── models/                    hand_landmarker.task goes here
├── vision/                    camera.py, hand_tracker.py, landmarks.py
├── gestures/
│   ├── gesture_types.py       GestureName, LifecycleState, ControlMode, GestureResult, GestureEvent
│   ├── features.py            HandFeatures, TwoHandFeatures, extractors
│   ├── static_gestures.py     Single-frame pose classification
│   ├── motion_gestures.py     SwipeDetector, DeltaTracker, ScalarDeltaTracker
│   ├── recognizer.py          GestureRecognizer interface + rule-based implementation
│   └── state_machine.py       Debouncing and lifecycle events
├── mappings/
│   ├── default_mappings.py    Mouse mode bindings
│   └── design_3d_mappings.py  3D mode overrides
├── actions/
│   ├── base.py                Action, ActionContext
│   ├── mouse.py               PointerController + mouse/orbit/zoom actions
│   ├── keyboard.py            Keyboard controller + KeyPress
│   └── registry.py            Name -> action lookup
├── app/                       application.py (orchestrator), state.py (cursor/scroll mapping, pause state)
├── config/                    settings.py (loads config.yaml)
├── ui/                        overlay.py (HUD and skeleton drawing)
├── utils/                     geometry.py, logging.py, smoothing.py
└── tests/
```

---

## Configuration

Settings live in `config.yaml` and are loaded by `config/settings.py`.

| Section | Keys | What they control |
|---|---|---|
| `camera` | `device`, `width`, `height`, `fps`, `mirrored` | Capture settings |
| `tracking` | `min_detection_confidence`, `min_tracking_confidence`, `max_num_hands` | MediaPipe settings. Use `max_num_hands: 2` for 3D two-hand control |
| `pinch` | `threshold`, `hold_time` | Pinch distance threshold, and seconds before a pinch becomes a hold |
| `swipe` | `min_distance`, `max_duration`, `min_velocity`, `cooldown` | Swipe detection |
| `gesture` | `min_confidence`, `static_hold_frames` | State machine debouncing |
| `cursor` | `sensitivity`, `deadzone`, `smoothing` | Pointer movement in mouse mode |
| `scroll` | `sensitivity`, `deadzone`, `invert` | Scroll speed and direction |
| `debug` | `enabled`, `show_landmarks` | On-screen debug lines and landmark dots |

3D feel is tuned in code for now:

| Constant | Where | Effect |
|---|---|---|
| `OrbitMove.SENSITIVITY` | `actions/mouse.py` | Pixels of mouse movement per unit of hand travel. Flip the sign to reverse direction |
| `Zoom.SENSITIVITY` | `actions/mouse.py` | Scroll notches per unit change in the distance between hands |
| `design_pinch_hold_time` | `Application(...)` | Seconds before a pinch starts an orbit in 3D mode (default 0.15) |
| `min_cutoff`, `beta` | `utils/smoothing.py` / `DeltaTracker` | Lower `min_cutoff` is smoother at rest. Higher `beta` has less lag when moving fast |

---

## Extending AirTouch

### Add a new gesture

1. Add a name to `GestureName` in `gestures/gesture_types.py`.
2. Detect it in `gestures/static_gestures.py` (a pose) or `gestures/motion_gestures.py` (a movement).
3. The state machine gives it a lifecycle automatically.
4. Bind it in a mapping table, e.g. `(GestureName.MY_GESTURE, LifecycleState.STARTED): ["MY_ACTION"]`.

### Add a new action

1. Subclass `Action` in `actions/`, set `name`, and implement `execute(context)`.
2. Register it in `Application._register_actions`.
3. If it needs per-frame data (deltas, positions), pass it through `ActionContext.extra` in `Application._build_context`.

### Support another 3D application or mapping set

Create a new mapping module like `mappings/design_3d_mappings.py`: a dictionary of overrides that falls back to the defaults for everything else. Only the gestures that behave differently need listing.

### Swap the input backend

Everything that moves the mouse goes through `PointerController` in `actions/mouse.py`. Implement that interface for another backend (for example a Blender add-on over a local socket) and the rest of the app doesn't change.

### Replace the recognizer

`GestureRecognizer` in `gestures/recognizer.py` takes `HandFeatures` and returns `GestureResult`. A learned model can implement the same interface and replace or augment the rules without touching features, the state machine or actions.

---

## Testing

`NullPointerController` and `NullKeyboardController` record calls instead of sending real input, so actions and the full pipeline can be tested without moving your mouse. Run the suite from the project root:

```powershell
pytest
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Unable to open file at models/hand_landmarker.task` | Download the model (step 2) and run from the project root |
| `W0000 ... inference_feedback_manager` / `NORM_RECT` warnings | Harmless MediaPipe warnings; ignore them |
| 3D mode does nothing in Blender | Click into the Blender viewport first so it has focus. Run AirTouch as administrator if Blender is |
| Two-hand zoom doesn't work | Set `tracking.max_num_hands: 2` and check the log for a "no 'hands' attribute" warning |
| Pinch flickers on and off | Raise `pinch_release_ratio`, or recalibrate `pinch.threshold` from logged `thumb_index_distance` |
| Orbit feels laggy or twitchy | Tune `min_cutoff` / `beta` and `OrbitMove.SENSITIVITY` |
| Swipes fire while you orbit | Make sure you are in 3D mode (HUD shows `| 3D`); swipes are suppressed while pinching there |
| Mouse stuck after a crash | Press and release the middle button once. This shouldn't happen: held buttons are released on shutdown |

---

## Roadmap

- Pan (Shift + middle-drag) and two-hand rotate (the twist angle is already computed)
- Per-application profiles with automatic switching by active window
- Blender add-on that sets the viewport directly over a local socket
- Learned gesture recognizer behind the existing `GestureRecognizer` interface

---

## License

MIT License