#!/usr/bin/env python3
"""
Gesture DJ by mtrxLabs
======================
Control music playback and system volume with hand gestures seen by a webcam.

Gestures
--------
1. VOLUME       Thumb + index extended, other three fingers curled ("pinch").
                Thumb-tip <-> index-tip distance, normalized by hand size
                (wrist -> middle-finger base), maps to system volume 0-100 %.
                Press 'c' to calibrate (pinch fully, then spread fully).
2. PLAY/PAUSE   Closed fist held >= 0.4 s toggles playback (media key).
3. NEXT/PREV    Open palm swiped sideways: wrist x-velocity above threshold.
                Swipe right (on screen) = next, left = previous.

Rules
-----
- The hand must be seen for MIN_CONSECUTIVE_FRAMES consecutive frames before
  any gesture acts (flicker protection).
- If the hand disappears, the last volume is held.
- Every action fires once (hold timers + cooldowns).

Audio path
----------
- System volume is set via pycaw (Windows master endpoint volume, i.e. the
  default output device).
- Play/pause + next/previous are sent as media keys via pyautogui, so they
  control whatever the OS media session is (browser, music app, ...).

Usage
-----
    python gesture_dj.py --camera 0
    python gesture_dj.py --no-mirror

Keys: q / ESC = quit, c = calibrate volume, m = toggle mirror, p = resync play state.

This file is intentionally a single file. Every value you may want to tune is
in the CONSTANTS block right below the imports.
"""

import argparse
import math
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# --- Optional dependencies: the app degrades gracefully if these fail ------
try:
    import pyautogui
    pyautogui.FAILSAFE = False     # don't slam the mouse into a corner to abort
    pyautogui.PAUSE = 0.0          # no artificial delay after each keypress
    HAVE_PYAUTOGUI = True
except Exception:                  # noqa: BLE001 - any import failure is fine
    HAVE_PYAUTOGUI = False

try:
    from ctypes import POINTER, cast

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    HAVE_PYCAW = True
except Exception:                  # noqa: BLE001
    HAVE_PYCAW = False


# ============================================================================
# TUNABLE CONSTANTS -- everything you might want to tweak lives here
# ============================================================================

# --- Camera / window --------------------------------------------------------
CAMERA_INDEX_DEFAULT = 0          # try 1, 2 ... if the wrong camera opens
FRAME_W_DEFAULT = 1280            # capture width  (bigger = sharper overlay)
FRAME_H_DEFAULT = 720             # capture height
CAPTURE_BACKEND = cv2.CAP_DSHOW   # Windows: DirectShow opens fast; MSMF lags
MIRROR_DEFAULT = True             # selfie view; toggle at runtime with 'm'
OVERLAY_WINDOW_NAME = "Gesture DJ"
OVERLAY_TOPMOST = True            # keep preview window above other windows

# --- MediaPipe Hands ----------------------------------------------------------
MAX_NUM_HANDS = 1                 # one hand is enough for all gestures
MODEL_COMPLEXITY = 0              # 0 = lite (fast), 1 = full (more accurate)
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.6

# --- Gesture gating -----------------------------------------------------------
MIN_CONSECUTIVE_FRAMES = 3        # hand must be seen this many frames in a row

# --- Volume gesture (pinch) ---------------------------------------------------
VOL_SMOOTHING_ALPHA = 0.4         # EMA weight of the newest raw value (0..1)
VOL_DEADZONE_PCT = 2.0            # ignore raw volume drift smaller than this
VOL_MIN_RATIO_DEFAULT = 0.35      # pinch distance / hand size  = 0 %  (pre-calib)
VOL_MAX_RATIO_DEFAULT = 1.65      # spread distance / hand size = 100 % (pre-calib)
CALIBRATION_SECONDS = 5.0         # how long 'c' records min/max pinch ratios
CALIB_MIN_SPAN = 0.4              # calibration only succeeds if spread - pinch >= this

# --- Fist gesture (play/pause) -------------------------------------------------
FIST_HOLD_S = 0.4                 # fist must be held this long to fire
FIST_COOLDOWN_S = 1.0             # ... then wait this long before it can fire again

# --- Swipe gesture (next/previous) ----------------------------------------------
SWIPE_OPEN_PALM_FINGERS = 4       # extended fingers (incl. thumb) for an open palm
SWIPE_VEL_THRESHOLD = 0.8         # wrist x-velocity in frame-widths per second
SWIPE_COOLDOWN_S = 1.0            # wait this long between swipes
SWIPE_WINDOW_S = 0.15             # history window used to estimate velocity

# --- Overlay look (OpenCV colors are BGR) ----------------------------------------
FONT = cv2.FONT_HERSHEY_DUPLEX
COLOR_WHITE = (245, 245, 245)
COLOR_GREEN = (90, 210, 110)
COLOR_YELLOW = (60, 200, 250)
COLOR_CYAN = (250, 220, 90)
COLOR_RED = (70, 70, 235)
COLOR_BAR = (90, 210, 110)
COLOR_BAR_BORDER = (245, 245, 245)
COLOR_BACKING = (28, 28, 28)
BAR_X = 48                        # volume bar left edge
BAR_W = 72                        # volume bar width
BAR_TOP = 140                     # volume bar top y
BAR_BOTTOM_MARGIN = 96            # volume bar bottom margin from frame bottom
PCT_TEXT_SCALE = 2.0              # big "NN %" text
LABEL_SCALE = 1.3                 # gesture label
STATE_SCALE = 1.0                 # play/pause state line
SMALL_SCALE = 0.75                # FPS / hints
ACTION_FLASH_S = 0.6              # how long the NEXT/BACK flash lasts
LANDMARK_THICKNESS = 3
LANDMARK_CIRCLE = 6
MAX_CAMERA_RETRIES = 45           # consecutive failed frame reads before exit

# --- Media key names (pyautogui) ---------------------------------------------------
KEY_PLAYPAUSE = "playpause"
KEY_NEXT = "nexttrack"
KEY_PREV = "prevtrack"

# --- Cached overlay objects (built once, reused every frame) ----------------------
_MP_DRAWING = mp.solutions.drawing_utils
_MP_HANDS = mp.solutions.hands
_SPEC_JOINT = _MP_DRAWING.DrawingSpec(color=(120, 220, 120),
                                      thickness=LANDMARK_THICKNESS,
                                      circle_radius=LANDMARK_CIRCLE)
_SPEC_BONE = _MP_DRAWING.DrawingSpec(color=(240, 240, 240),
                                     thickness=LANDMARK_THICKNESS)
_PLAY_TEXT = "PLAYING (app belief)"
_PAUSE_TEXT = "PAUSED (app belief)"
_PLAY_TW = cv2.getTextSize(_PLAY_TEXT, FONT, STATE_SCALE, 2)[0][0]
_PAUSE_TW = cv2.getTextSize(_PAUSE_TEXT, FONT, STATE_SCALE, 2)[0][0]

# ============================================================================
# MediaPipe landmark indices (0-20). Only the ones we use are named.
# ============================================================================
WRIST = 0
THUMB_IP = 3
THUMB_TIP = 4
INDEX_PIP = 6
INDEX_TIP = 8
INDEX_MCP = 5
MIDDLE_PIP = 10
MIDDLE_TIP = 12
MIDDLE_MCP = 9          # "middle-finger base" -- used as the hand-size anchor
RING_PIP = 14
RING_TIP = 16
RING_MCP = 13
PINKY_PIP = 18
PINKY_TIP = 20
PINKY_MCP = 17          # reference point for the thumb extension test


def _dist(a, b):
    """Euclidean distance between two (x, y) pixel points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _finger_extended(pts, tip, pip):
    """A finger counts as extended when its tip is farther from the wrist than
    its PIP joint. Using distances (instead of raw y coordinates) keeps this
    working at any hand rotation."""
    return _dist(pts[tip], pts[WRIST]) > _dist(pts[pip], pts[WRIST]) * 1.05


def _thumb_extended(pts):
    """The thumb is 'extended' when its tip is farther from the pinky base than
    its IP joint is -- i.e. it sticks out sideways instead of lying on the palm."""
    return _dist(pts[THUMB_TIP], pts[PINKY_MCP]) > _dist(pts[THUMB_IP], pts[PINKY_MCP]) * 1.05


# ============================================================================
# System volume (pycaw)
# ============================================================================
class VolumeController:
    """Sets the Windows master (endpoint) volume. Write-only: the app never
    reads the volume back; its own 0-100 value is the source of truth."""

    def __init__(self):
        self._volume = None
        if not HAVE_PYCAW:
            print("[warn] pycaw not available -- volume will be shown but NOT applied")
            return
        try:
            device = AudioUtilities.GetSpeakers()
            interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self._volume = cast(interface, POINTER(IAudioEndpointVolume))
        except Exception as exc:  # noqa: BLE001 - no audio endpoint, etc.
            print(f"[warn] could not open the audio endpoint ({exc}); volume disabled")

    @property
    def available(self):
        return self._volume is not None

    def set_percent(self, pct):
        if self._volume is not None:
            self._volume.SetMasterVolumeLevelScalar(max(0, min(100, pct)) / 100.0, None)


# ============================================================================
# Media keys (pyautogui)
# ============================================================================
_warned_media = {"done": False}


def send_media_key(key_name):
    """Press a media key. Warns once (not every frame) if unavailable."""
    if not HAVE_PYAUTOGUI:
        if not _warned_media["done"]:
            print("[warn] pyautogui not available -- media keys are disabled")
            _warned_media["done"] = True
        return
    try:
        pyautogui.press(key_name)
    except Exception as exc:  # noqa: BLE001
        if not _warned_media["done"]:
            print(f"[warn] media key '{key_name}' failed: {exc}")
            _warned_media["done"] = True


# ============================================================================
# Gesture engine -- pure logic, no I/O
# ============================================================================
class HandGestureEngine:
    """Consumes per-frame hand landmarks and emits events:
        ('volume', pct) | ('toggle',) | ('next',) | ('prev',)
    Holds internal state machines for calibration, fist hold and swipes."""

    def __init__(self):
        self.seen_consec = 0
        self.applied_pct = None            # last volume actually applied (None = never)
        self.smooth_ratio = None           # EMA-smoothed pinch distance ratio
        self.calib_min = VOL_MIN_RATIO_DEFAULT
        self.calib_max = VOL_MAX_RATIO_DEFAULT
        self.calibrating_until = None
        self.calib_obs_min = None
        self.calib_obs_max = None
        self.calib_msg = ""
        self.fist_since = None
        self.fist_fired = False
        self.fist_cooldown_until = 0.0
        self.swipe_pts = deque()           # (timestamp, wrist x / frame width)
        self.swipe_cooldown_until = 0.0
        # Overlay helpers (read by the drawing code):
        self.gesture_label = "SHOW A HAND"
        self.pinch_px = None               # (midpoint, px distance) for the overlay
        self.hand_size_px = None

    # -- calibration ----------------------------------------------------------
    def start_calibration(self, now):
        self.calibrating_until = now + CALIBRATION_SECONDS
        self.calib_obs_min = None
        self.calib_obs_max = None
        self.calib_msg = ""
        print(f"[info] calibrating volume for {CALIBRATION_SECONDS:.0f} s: "
              f"pinch fully, then spread fully")

    def _finish_calibration(self, now):
        self.calibrating_until = None
        lo, hi = self.calib_obs_min, self.calib_obs_max
        if lo is not None and hi is not None and (hi - lo) >= CALIB_MIN_SPAN:
            self.calib_min, self.calib_max = lo, hi
            self.calib_msg = f"CALIBRATED {self.calib_min:.2f}-{self.calib_max:.2f}"
            print(f"[info] calibration done: pinch={lo:.2f} spread={hi:.2f}")
        else:
            self.calib_msg = "CALIB FAILED: use full pinch + full spread"
            print("[warn] calibration failed -- need both a full pinch and a full spread")

    def _calibration_tick(self, ratio):
        if self.calib_obs_min is None or ratio < self.calib_obs_min:
            self.calib_obs_min = ratio
        if self.calib_obs_max is None or ratio > self.calib_obs_max:
            self.calib_obs_max = ratio

    # -- main update ------------------------------------------------------------
    def update(self, pts, has_hand, now, frame_w):
        """pts: list of 21 (x, y) pixel tuples (empty list when no hand)."""
        events = []

        if self.calibrating_until is not None and now >= self.calibrating_until:
            self._finish_calibration(now)

        if not has_hand:
            self.seen_consec = 0
            self.fist_since = None
            self.fist_fired = False
            self.swipe_pts.clear()
            self.pinch_px = None
            self.gesture_label = "NO HAND - VOLUME HELD"
            return events

        self.seen_consec += 1
        if self.seen_consec < MIN_CONSECUTIVE_FRAMES:
            self.gesture_label = "HAND ACQUIRED..."
            self.pinch_px = None
            return events

        cls, _ext, hand_size = self._classify(pts)
        self.hand_size_px = hand_size

        if cls == "volume":
            self._on_volume(pts, hand_size, now, events)
        elif cls == "fist":
            self._on_fist(now, events)
        elif cls == "swipe":
            self._on_swipe(pts, now, frame_w, events)
        else:
            self.gesture_label = "..."
            self.pinch_px = None

        # Leaving the fist gesture re-arms it.
        if cls != "fist":
            self.fist_since = None
            self.fist_fired = False

        return events

    # -- classification -----------------------------------------------------------
    def _classify(self, pts):
        thumb = _thumb_extended(pts)
        index = _finger_extended(pts, INDEX_TIP, INDEX_PIP)
        middle = _finger_extended(pts, MIDDLE_TIP, MIDDLE_PIP)
        ring = _finger_extended(pts, RING_TIP, RING_PIP)
        pinky = _finger_extended(pts, PINKY_TIP, PINKY_PIP)
        ext = [thumb, index, middle, ring, pinky]
        hand_size = max(_dist(pts[WRIST], pts[MIDDLE_MCP]), 1e-3)

        if ext[0] and ext[1] and not ext[2] and not ext[3] and not ext[4]:
            return "volume", ext, hand_size        # thumb + index "pinch"
        if not ext[1] and not ext[2] and not ext[3] and not ext[4]:
            return "fist", ext, hand_size          # 4 fingers curled (thumb ignored)
        if sum(ext) >= SWIPE_OPEN_PALM_FINGERS:
            return "swipe", ext, hand_size         # open palm
        return "other", ext, hand_size

    # -- gesture handlers -------------------------------------------------------------
    def _on_volume(self, pts, hand_size, now, events):
        ratio = _dist(pts[THUMB_TIP], pts[INDEX_TIP]) / hand_size
        dist_px = _dist(pts[THUMB_TIP], pts[INDEX_TIP])
        self.pinch_px = (pts[THUMB_TIP], pts[INDEX_TIP], dist_px)

        if self.calibrating_until is not None:
            self._calibration_tick(ratio)
            remain = self.calibrating_until - now
            self.gesture_label = f"CALIBRATING {remain:0.0f}s - pinch then spread"
            return

        # Exponential smoothing keeps the bar from twitching.
        if self.smooth_ratio is None:
            self.smooth_ratio = ratio
        else:
            self.smooth_ratio += VOL_SMOOTHING_ALPHA * (ratio - self.smooth_ratio)

        span = max(self.calib_max - self.calib_min, 1e-3)
        val = (self.smooth_ratio - self.calib_min) / span
        val = max(0.0, min(1.0, val))
        pct = round(val * 100.0)

        # Small deadzone: raw drift below VOL_DEADZONE_PCT never reaches the output.
        if self.applied_pct is not None and abs(pct - self.applied_pct) < VOL_DEADZONE_PCT:
            self.gesture_label = "PINCH - VOLUME"
            return
        self.applied_pct = pct
        self.gesture_label = "PINCH - VOLUME"
        events.append(("volume", pct))

    def _on_fist(self, now, events):
        self.pinch_px = None
        if self.fist_since is None:
            self.fist_since = now
        progress = now - self.fist_since
        if not self.fist_fired and progress >= FIST_HOLD_S and now >= self.fist_cooldown_until:
            events.append(("toggle",))
            self.fist_fired = True
            self.fist_cooldown_until = now + FIST_COOLDOWN_S
            self.gesture_label = "FIST - TOGGLED"
            return
        if self.fist_fired:
            self.gesture_label = "FIST - COOLDOWN"
        else:
            self.gesture_label = f"FIST - HOLD {progress:.1f}/{FIST_HOLD_S:.1f}s"

    def _on_swipe(self, pts, now, frame_w, events):
        self.pinch_px = None
        self.swipe_pts.append((now, pts[WRIST][0] / max(frame_w, 1)))
        while self.swipe_pts and now - self.swipe_pts[0][0] > SWIPE_WINDOW_S:
            self.swipe_pts.popleft()

        self.gesture_label = "OPEN PALM - SWIPE L/R"
        if now < self.swipe_cooldown_until or len(self.swipe_pts) < 2:
            return
        (t0, x0), (t1, x1) = self.swipe_pts[0], self.swipe_pts[-1]
        dt = t1 - t0
        if dt <= 0:
            return
        vel = (x1 - x0) / dt                       # frame-widths per second
        if abs(vel) >= SWIPE_VEL_THRESHOLD:
            events.append(("next",) if vel > 0 else ("prev",))
            self.swipe_cooldown_until = now + SWIPE_COOLDOWN_S
            self.swipe_pts.clear()                 # one swipe = one action


# ============================================================================
# Overlay drawing -- big, clean, readable on a screen recording
# ============================================================================
def draw_text(img, text, org, scale, color, thickness=1, center_x=None, backing=True):
    """putText with an optional dark backing plate and optional centering."""
    (w, h), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    if center_x is not None:
        x = int(center_x - w / 2)
    if backing:
        pad = 6
        cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + base + pad),
                      COLOR_BACKING, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
    return x, w  # useful when an icon must sit next to the text


def draw_volume_bar(img, pct):
    """Large vertical bar with ticks, plus a big percentage readout."""
    h_img = img.shape[0]
    x, w = BAR_X, BAR_W
    y_bot = h_img - BAR_BOTTOM_MARGIN
    y_top = BAR_TOP
    bar_h = y_bot - y_top
    cv2.rectangle(img, (x - 8, y_top - 8), (x + w + 8, y_bot + 8), COLOR_BACKING, -1)
    cv2.rectangle(img, (x, y_top), (x + w, y_bot), COLOR_BAR_BORDER, 3)
    fill_h = int(bar_h * max(0, min(100, pct)) / 100.0)
    if fill_h > 0:
        cv2.rectangle(img, (x, y_bot - fill_h), (x + w, y_bot), COLOR_BAR, -1)
    for tick in (0, 25, 50, 75, 100):  # tick marks on the right edge
        ty = y_bot - int(bar_h * tick / 100)
        cv2.line(img, (x + w, ty), (x + w + 12, ty), COLOR_BAR_BORDER, 2)
        draw_text(img, str(tick), (x + w + 18, ty + 8), 0.55, COLOR_WHITE, 1, backing=False)
    draw_text(img, f"{max(0, min(100, pct)):3d}%", (x + w + 64, y_top + 84),
              PCT_TEXT_SCALE, COLOR_WHITE, 3)


def draw_play_icon(img, playing, x, y, size=26):
    """Filled triangle for playing, two bars for paused; drawn left of a text."""
    if playing:
        cv2.fillPoly(img, [np.array([[x, y], [x, y + size], [x + int(size * 0.9), y + size // 2]])],
                     COLOR_GREEN)
    else:
        bar_w = max(6, size // 4)
        cv2.rectangle(img, (x, y), (x + bar_w, y + size), COLOR_YELLOW, -1)
        cv2.rectangle(img, (x + 2 * bar_w, y), (x + 3 * bar_w, y + size), COLOR_YELLOW, -1)


def draw_action_flash(img, text, fraction):
    """Full-screen flash when next/previous fires; fraction fades it out."""
    if fraction <= 0:
        return
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), COLOR_BACKING, -1)
    cv2.addWeighted(overlay, 0.45 * fraction, img, 1.0 - 0.45 * fraction, 0, img)
    h_img, w_img = img.shape[:2]
    draw_text(img, text, (0, h_img // 2 + 40), 3.2, COLOR_YELLOW, 8, center_x=w_img // 2)
    draw_text(img, "media key sent", (0, h_img // 2 + 110), 0.9, COLOR_WHITE, 1,
              center_x=w_img // 2)


def draw_landmarks(img, hand_landmarks, pinch):
    """Draw the MediaPipe skeleton and, while pinching, the measured distance."""
    _MP_DRAWING.draw_landmarks(img, hand_landmarks, _MP_HANDS.HAND_CONNECTIONS,
                               _SPEC_JOINT, _SPEC_BONE)
    if pinch is not None:
        tip_a, tip_b, dist_px = pinch
        cv2.line(img, tip_a, tip_b, COLOR_YELLOW, 2)
        mid = ((tip_a[0] + tip_b[0]) // 2, (tip_a[1] + tip_b[1]) // 2)
        cv2.circle(img, mid, 10, COLOR_YELLOW, 2)
        draw_text(img, f"{dist_px:.0f}px", (mid[0] + 16, mid[1] - 16), 0.7,
                  COLOR_YELLOW, 1, backing=False)


# ============================================================================
# Main
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Gesture DJ by mtrxLabs -- webcam hand gestures for music "
                    "control (volume + media keys).")
    p.add_argument("--camera", type=int, default=CAMERA_INDEX_DEFAULT,
                   help=f"camera index (default {CAMERA_INDEX_DEFAULT})")
    p.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=MIRROR_DEFAULT,
                   help="mirror the camera view (default: on; 'm' toggles live)")
    p.add_argument("--width", type=int, default=FRAME_W_DEFAULT,
                   help=f"capture width (default {FRAME_W_DEFAULT})")
    p.add_argument("--height", type=int, default=FRAME_H_DEFAULT,
                   help=f"capture height (default {FRAME_H_DEFAULT})")
    return p.parse_args()


def open_camera(args):
    cap = cv2.VideoCapture(args.camera, CAPTURE_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"[error] could not open camera index {args.camera}. "
              f"Try another index: python gesture_dj.py --camera 1 (or 2, 3 ...)")
        sys.exit(2)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[info] camera {args.camera} opened at {w}x{h}")
    return cap


def main():
    args = parse_args()

    print("=" * 64)
    print("Gesture DJ by mtrxLabs -- hand gestures -> music control")
    print(f"  pycaw volume:    {'yes' if HAVE_PYCAW else 'NO (volume not applied)'}")
    print(f"  pyautogui keys:  {'yes' if HAVE_PYAUTOGUI else 'NO (media keys off)'}")
    print("  keys: q quit | c calibrate | m mirror | p resync play state")
    print("=" * 64)

    volume = VolumeController()
    engine = HandGestureEngine()

    cap = open_camera(args)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE)

    cv2.namedWindow(OVERLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
    if OVERLAY_TOPMOST:
        try:
            cv2.setWindowProperty(OVERLAY_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        except Exception as exc:  # noqa: BLE001 - older OpenCV builds
            print(f"[info] keeping the window non-topmost ({exc})")

    playing = False          # app-side play/pause BELIEF (can drift -- see README).
                             # If overlay is opposite vs your player, press 'p' once to resync
                             # (flips label only, sends no media key).
    flash_text = ""
    flash_until = 0.0
    fps = 0.0
    prev_t = time.monotonic()
    failed_reads = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failed_reads += 1
                if failed_reads > MAX_CAMERA_RETRIES:
                    print("[error] camera stopped delivering frames -- exiting")
                    break
                continue
            failed_reads = 0

            if args.mirror:
                frame = cv2.flip(frame, 1)

            # --- perception ------------------------------------------------
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            hand_lms = None
            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]

            h_img, w_img = frame.shape[:2]
            now = time.monotonic()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = fps * 0.9 + (1.0 / dt) * 0.1

            pts = []
            if hand_lms is not None:
                pts = [(int(lm.x * w_img), int(lm.y * h_img)) for lm in hand_lms.landmark]

            events = engine.update(pts, hand_lms is not None, now, w_img)

            # --- act on events ------------------------------------------------
            for event in events:
                if event[0] == "volume":
                    pct = event[1]
                    volume.set_percent(pct)
                elif event[0] == "toggle":
                    playing = not playing
                    send_media_key(KEY_PLAYPAUSE)
                    print(f"[event] play/pause -> {'PLAYING' if playing else 'PAUSED'} (belief)")
                elif event[0] == "next":
                    send_media_key(KEY_NEXT)
                    flash_text, flash_until = "NEXT >>", now + ACTION_FLASH_S
                    print("[event] next track")
                elif event[0] == "prev":
                    send_media_key(KEY_PREV)
                    flash_text, flash_until = "<< BACK", now + ACTION_FLASH_S
                    print("[event] previous track")

            # --- overlay -----------------------------------------------------
            if hand_lms is not None:
                draw_landmarks(frame, hand_lms, engine.pinch_px)

            if engine.calibrating_until is not None:
                lo = engine.calib_obs_min
                hi = engine.calib_obs_max
                if lo is not None:
                    draw_text(frame, f"pinch {lo:.2f}   spread "
                                     f"{(hi if hi is not None else lo):.2f}",
                              (0, 148), 0.8, COLOR_YELLOW, 1,
                              center_x=w_img // 2)
            else:
                draw_text(frame, engine.calib_msg, (0, 148), 0.8,
                          COLOR_GREEN if "CALIBRATED" in engine.calib_msg else COLOR_YELLOW,
                          1, center_x=w_img // 2)

            draw_volume_bar(frame, engine.applied_pct if engine.applied_pct is not None else 0)

            # gesture label (top center, big)
            draw_text(frame, engine.gesture_label, (0, 58), LABEL_SCALE,
                      COLOR_CYAN, 2, center_x=w_img // 2)

            # play state + icon (below the gesture label)
            state_text = _PLAY_TEXT if playing else _PAUSE_TEXT
            state_color = COLOR_GREEN if playing else COLOR_YELLOW
            tw = _PLAY_TW if playing else _PAUSE_TW
            x_text = int(w_img // 2 - (tw + 40) / 2)
            draw_play_icon(frame, playing, x_text, 96, 26)
            draw_text(frame, state_text, (x_text + 40, 100 + 12), STATE_SCALE,
                      state_color, 2, backing=False)

            # FPS (top left)
            draw_text(frame, f"FPS: {fps:4.1f}", (16, 40), SMALL_SCALE, COLOR_WHITE, 1)

            # bottom hint strip
            cv2.rectangle(frame, (0, h_img - 40), (w_img, h_img), COLOR_BACKING, -1)
            draw_text(frame, "q = quit    c = calibrate    m = mirror    p = resync play state",
                      (0, h_img - 12), 0.8, COLOR_WHITE, 1, center_x=w_img // 2)

            # next/previous flash on top of everything
            if now < flash_until:
                draw_action_flash(frame, flash_text, (flash_until - now) / ACTION_FLASH_S)

            cv2.imshow(OVERLAY_WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                engine.start_calibration(now)
            elif key == ord("m"):
                args.mirror = not args.mirror
            elif key == ord("p"):
                playing = not playing
                print(f"[info] play-state belief resynced -> {'PLAYING' if playing else 'PAUSED'} (no key sent)")
    except KeyboardInterrupt:
        print("\n[info] interrupted -- shutting down")
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[info] bye")


if __name__ == "__main__":
    main()
