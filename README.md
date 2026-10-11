# Gesture DJ

Control music playback and volume with **hand gestures** seen by a webcam.
Software only, runs locally: no cloud APIs, no accounts, no hardware required.

- **Volume** — pinch thumb + index, spread / close them.
- **Play / Pause** — hold a closed fist for 0.4 s.
- **Next / Previous** — swipe an open palm sideways (right = next, left = previous).
- Sets **system volume** (pycaw) and sends **media keys** (pyautogui), so it works
  with whatever is playing: browser, music app, video player, etc.

On-screen overlay shows a large volume bar with percentage, gesture label,
play state, next/back flashes, and FPS — readable on a screen recording.

```text
release/
├─ gesture_dj.py      # the whole app (single file, commented)
├─ requirements.txt   # pinned dependencies (Python 3.10–3.12)
├─ README.md          # this file
├─ LICENSE            # MIT
└─ .gitignore         # venv / caches / OS junk
```

## Requirements

- Windows 10/11 + webcam
- **Python 3.10–3.12** (MediaPipe has no 3.13 wheels yet)
- Packages in `requirements.txt`: `mediapipe`, `opencv-python`, `numpy==1.26.4`,
  `pycaw`, `comtypes`, `pyautogui`

## Setup

```powershell
cd gesture-dj
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Run

```powershell
.venv\Scripts\python gesture_dj.py
.venv\Scripts\python gesture_dj.py --camera 1
.venv\Scripts\python gesture_dj.py --no-mirror
.venv\Scripts\python gesture_dj.py --width 960 --height 540
```

| Option | Meaning |
|--------|---------|
| `--camera INDEX` | webcam index (default 0; try 1, 2 … if the wrong camera opens) |
| `--mirror / --no-mirror` | mirrored selfie view (default on) |
| `--width W --height H` | capture resolution (default 1280×720) |

## Gestures

| Gesture | Action | Details |
|---------|--------|---------|
| **Pinch** (thumb + index out, other fingers curled) | volume 0–100 % | thumb↔index distance / hand size. Smoothed + deadzone. |
| `c` key | calibrate volume | 5 s: fully **pinch**, then fully **spread**. Redo when lighting / seating changes. |
| **Closed fist** held ≥ 0.4 s | play / pause | fires once, 1 s cooldown. |
| **Open palm swipe** → / ← | next / previous | wrist x-velocity trigger, 1 s cooldown. |
| Hand leaves frame | — | volume held, nothing fires. |

Keys: `q` / `ESC` quit · `c` calibrate · `m` mirror · `p` resync play state.

## Play-state note (honest caveat)

The app tracks play/pause as its own toggle — it never reads the player.
If the overlay says PLAYING while the player is paused (or vice versa),
press `p` once to resync the label. It sends no media key, only flips the
display. The fist gesture itself always toggles the real player.

## Troubleshooting

- **Wrong / black camera:** try `--camera 1`, `--camera 2`, … (laptops often
  expose an IR or virtual cam at 0). Uses DirectShow (`CAP_DSHOW`).
- **Install fails on Python 3.13:** use `py -3.12`. Keep `numpy==1.26.4`
  (mediapipe 0.10.x targets the numpy 1.x ABI).
- **Volume moves the wrong device:** set your speaker as default output
  (Settings → System → Sound). Media keys are unaffected.
- **Twitchy volume:** press `c` to recalibrate, improve lighting, keep the
  whole hand in frame (needs 3 consecutive frames).
- **Low FPS:** lower `--width/--height` (e.g. 960×540), close other camera apps.
  `MODEL_COMPLEXITY = 0` in `gesture_dj.py` is already the fast model.

All tunables (`FIST_HOLD_S`, `SWIPE_VEL_THRESHOLD`, smoothing, deadzone, …)
live in the constants block at the top of `gesture_dj.py`.

---
Built by mtrxLabs.
