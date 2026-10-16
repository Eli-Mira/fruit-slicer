"""Webcam hand-gesture detection engine.

Detection only — no system actions live here. Run directly for a debug
window, or import and poll from another script:

    from gesture_engine import get_gesture, get_hand_position, stop

A daemon thread owns the camera; polling never blocks the caller.
Gesture labels: fist, open_palm, thumbs_up, pinch,
swipe_left/right/up/down, or None.
"""

from __future__ import annotations

import atexit
import collections
import os
import threading
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- tunables -------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

HISTORY_LEN = 10        # palm positions kept for swipe analysis
SWIPE_MIN_SAMPLES = 6   # history must refill after a swipe before the next one
SWIPE_MIN_TRAVEL = 0.16 # normalized palm travel inside the window to count
SWIPE_AXIS_MARGIN = 1.25  # dominant axis must beat secondary by this factor
SWIPE_TTL = 0.30        # seconds a fired swipe label stays readable

PINCH_MAX_RATIO = 0.45  # thumb-index tip distance / hand size
EXTEND_RATIO = 1.12     # tip-vs-pip wrist-distance ratio for "finger extended"
THUMBS_UP_MARGIN = 0.55 # thumb tip height above thumb MCP / hand size

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "hand_landmarker.task")

_WRIST, _THUMB_TIP, _INDEX_TIP = 0, 4, 8
_THUMB_MCP, _THUMB_IP = 2, 3
_INDEX_MCP, _MIDDLE_MCP, _RING_MCP, _PINKY_MCP = 5, 9, 13, 17
_FINGER_PIPS = {8: 6, 12: 10, 16: 14, 20: 18}
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# --- shared state ----------------------------------------------------------
_LOCK = threading.Lock()
_START_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None
_STOP = threading.Event()
_CAMERA_OK = False
_STATIC_GESTURE: str | None = None
_SWIPE: tuple[str, float] | None = None   # (label, expires_at)
_HAND_POS: tuple[float, float] | None = None
_LANDMARKS: list[tuple[int, int]] | None = None
_FRAME_BUFFERS = [None, None]
_FRAME_INDEX = 0
_FPS = 0.0


def _dist(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _extended(lm, tip: int, pip: int, wrist) -> bool:
    return _dist(lm[tip], wrist) > _dist(lm[pip], wrist) * EXTEND_RATIO


def _classify_static(lm) -> str | None:
    """Pose classification for a single frame of landmarks."""
    wrist = lm[_WRIST]
    hand_size = max(_dist(lm[_MIDDLE_MCP], wrist), 1e-6)

    thumb_out = (_dist(lm[_THUMB_TIP], lm[_PINKY_MCP])
                 > _dist(lm[_THUMB_IP], lm[_PINKY_MCP]) * 1.08)
    four_extended = sum(_extended(lm, t, p, wrist) for t, p in _FINGER_PIPS.items())

    pinch_ratio = _dist(lm[_THUMB_TIP], lm[_INDEX_TIP]) / hand_size
    if pinch_ratio < PINCH_MAX_RATIO:
        return "pinch"
    if (thumb_out and four_extended == 0
            and (lm[_THUMB_MCP].y - lm[_THUMB_TIP].y) > hand_size * THUMBS_UP_MARGIN):
        return "thumbs_up"
    if four_extended == 0:
        return "fist"
    if four_extended >= 4:
        return "open_palm"
    return None


def _detect_swipe(history: collections.deque) -> str | None:
    if len(history) < SWIPE_MIN_SAMPLES:
        return None
    (_, x0, y0), (_, x1, y1) = history[0], history[-1]
    dx, dy = x1 - x0, y1 - y0
    if abs(dx) >= abs(dy) * SWIPE_AXIS_MARGIN and abs(dx) >= SWIPE_MIN_TRAVEL:
        return "swipe_right" if dx > 0 else "swipe_left"
    if abs(dy) > abs(dx) * SWIPE_AXIS_MARGIN and abs(dy) >= SWIPE_MIN_TRAVEL:
        return "swipe_up" if dy < 0 else "swipe_down"  # image y grows downward
    return None


MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 2.0  # seconds between reconnect attempts

# ---------------------------------------------------------------------------
# Camera discovery – try a small list of device indices and back‑ends,
# validating that we can actually read a frame before we claim the camera.
# ---------------------------------------------------------------------------
_PREFERRED_BACKENDS = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
_PREFERRED_INDEXES = list(range(5))  # 0,1,2,3,4


def _try_open(index: int, backend: int):
    """Open VideoCapture at *index* with *backend*; return (cap, ret, frame) or (None, None, None)."""
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None, None, None
    # Read a single frame to confirm the camera actually works.
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return None, None, None
    # Keep the open capture — set resolution now so the engine uses it.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    return cap, ret, frame


def discover_camera():
    """Return a VideoCapture that works, or None.

    Tries each preferred index with each preferred backend; the first
    combination that can successfully read a frame is returned.
    """
    for idx in _PREFERRED_INDEXES:
        for backend in _PREFERRED_BACKENDS:
            cap, _, _ = _try_open(idx, backend)
            if cap is not None:
                # Successfully opened and read a frame – keep this cap.
                return cap
    return None


def _open_camera():
    """Open and configure the capture device using the discovery routine.

    Returns a VideoCapture that is known to produce frames, or None on failure.
    """
    cap = discover_camera()
    if cap is None:
        return None
    # The discovery routine already set the desired resolution; just ensure
    # the internal flags are applied (they were set during the trial open).
    return cap


def _loop() -> None:
    global _CAMERA_OK, _STATIC_GESTURE, _SWIPE, _HAND_POS, _LANDMARKS, _FRAME_BUFFERS, _FRAME_INDEX, _FPS

    def open_camera():
        cap = _open_camera()
        if cap is not None:
            with _LOCK:
                _CAMERA_OK = True
            print(f"[gesture_engine] Camera opened successfully (index={cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)})")
        return cap


    cap = open_camera()
    if cap is None:
        print(f"[gesture_engine] ERROR: cannot open any camera on indexes 0‑4 with backends {_PREFERRED_BACKENDS}. "
              "Check that a webcam is connected and that Windows camera privacy "
              "settings allow desktop apps to access it.")
        with _LOCK:
            _CAMERA_OK = False
        return

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        landmarker = vision.HandLandmarker.create_from_options(options)
    except Exception as exc:
        print(f"[gesture_engine] ERROR: failed to create HandLandmarker: {exc!r}")
        try:
            cap.release()
        except Exception:
            pass
        with _LOCK:
            _CAMERA_OK = False
        return

    history: collections.deque = collections.deque(maxlen=HISTORY_LEN)
    frames_done, fps_t0 = 0, time.monotonic()
    reconnect_attempts = 0

    try:
        while not _STOP.is_set():
            ok, frame = cap.read()
            if not ok:
                print("[gesture_engine] WARNING: camera read failed, attempting reconnect...")
                with _LOCK:
                    _CAMERA_OK = False

                # Release the old capture before attempting a new one.
                try:
                    cap.release()
                except Exception:
                    pass

                # Reconnect – try to rediscover a working camera.
                while reconnect_attempts < MAX_RECONNECT_ATTEMPTS and not _STOP.is_set():
                    reconnect_attempts += 1
                    print(f"[gesture_engine] Reconnect attempt "
                          f"{reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS} "
                          f"in {RECONNECT_DELAY}s...")
                    time.sleep(RECONNECT_DELAY)
                    cap = discover_camera()
                    if cap is not None:
                        reconnect_attempts = 0
                        # After successful reconnect, reset the ok flag; the next loop iteration will read fresh.
                        with _LOCK:
                            _CAMERA_OK = True
                        break

                if cap is None or _STOP.is_set():
                    print("[gesture_engine] ERROR: max reconnect attempts reached, giving up.")
                    break
                # Continue the outer while loop to read from the new capture.
                continue

            # ---- frame succeeded -------------------------------------------------
            frame = cv2.flip(frame, 1)  # selfie view: matches user perception
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # VIDEO mode requires strictly monotonically increasing timestamps.
            # Use monotonic wall‑clock ms (not CAP_PROP_POS_MSEC, which can
            # reset or flatline and breaks landmark tracking).
            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                timestamp_ms,
            )

            static, pos, pts, swipe = None, None, None, None
            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                static = _classify_static(lm)
                cx = sum(lm[i].x for i in
                         (_WRIST, _INDEX_MCP, _MIDDLE_MCP, _RING_MCP, _PINKY_MCP)) / 5.0
                cy = sum(lm[i].y for i in
                         (_WRIST, _INDEX_MCP, _MIDDLE_MCP, _RING_MCP, _PINKY_MCP)) / 5.0
                pos = (cx, cy)
                history.append((time.monotonic(), cx, cy))
                swipe = _detect_swipe(history)
                if swipe:
                    history.clear()  # require fresh travel before the next swipe
                h, w = frame.shape[:2]
                pts = [(int(p.x * w), int(p.y * h)) for p in lm]
            else:
                history.clear()  # no hand = reset swipe history

            now = time.monotonic()
            with _LOCK:
                if swipe is not None:
                    _SWIPE = (swipe, now + SWIPE_TTL)
                elif _SWIPE is not None and now >= _SWIPE[1]:
                    _SWIPE = None
                _STATIC_GESTURE = static
                _HAND_POS = pos
                _LANDMARKS = pts
                # Double‑buffer: write to back buffer, then atomically swap index
                back_idx = 1 - _FRAME_INDEX
                _FRAME_BUFFERS[back_idx] = frame
                _FRAME_INDEX = back_idx  # atomic in CPython (GIL)
                # Exponential moving average for FPS (smoother, no counter reset jitter)
                dt = now - fps_t0
                if dt > 0:
                    inst_fps = 1.0 / dt
                    _FPS = 0.9 * _FPS + 0.1 * inst_fps
                fps_t0 = now
    except Exception as exc:  # never let the thread die silently
        print(f"[gesture_engine] detection loop crashed: {exc!r}")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            landmarker.close()
        except Exception:
            pass
        with _LOCK:
            _CAMERA_OK = False
            _STATIC_GESTURE = None
            _HAND_POS = None
            _LANDMARKS = None
            _FRAME_BUFFERS[0] = None
            _FRAME_BUFFERS[1] = None
            _SWIPE = None


def _ensure_started() -> None:
    global _THREAD
    if _THREAD is not None and _THREAD.is_alive():
        return
    with _START_LOCK:
        if _THREAD is not None and _THREAD.is_alive():
            return
        if not os.path.isfile(_MODEL_PATH):
            raise FileNotFoundError(
                f"hand_landmarker.task not found at {_MODEL_PATH}. Download it from "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/latest/hand_landmarker.task")
        _STOP.clear()
        _THREAD = threading.Thread(target=_loop, name="gesture-engine", daemon=True)
        _THREAD.start()


def get_gesture() -> str | None:
    """Latest gesture label, or None if no clear gesture right now.

    Static poses persist while held; swipes stay readable for ~0.3 s so a
    slower poller can still catch them.
    """
    _ensure_started()
    with _LOCK:
        if _SWIPE is not None and time.monotonic() < _SWIPE[1]:
            return _SWIPE[0]
        return _STATIC_GESTURE


def get_hand_position() -> tuple[float, float] | None:
    """Normalized (x, y) palm center in [0..1], or None if no hand."""
    _ensure_started()
    with _LOCK:
        return _HAND_POS


def is_running() -> bool:
    with _LOCK:
        return _CAMERA_OK


def stop() -> None:
    """Release the camera and join the detection thread."""
    global _THREAD
    _STOP.set()
    thread, _THREAD = _THREAD, None
    if thread is not None and thread.is_alive():
        thread.join(timeout=3.0)


def _read_gesture_unlocked(now: float) -> str | None:
    """Read gesture while _LOCK is already held — avoids re-entrant lock deadlock."""
    if _SWIPE is not None and now < _SWIPE[1]:
        return _SWIPE[0]
    return _STATIC_GESTURE


def run_debug() -> None:
    """Live camera window with landmarks drawn and the current gesture shown."""
    _ensure_started()
    blank = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    try:
        while not _STOP.is_set():
            with _LOCK:
                frame = None if _FRAME_BUFFERS[_FRAME_INDEX] is None else _FRAME_BUFFERS[_FRAME_INDEX].copy()
                pts = _LANDMARKS
                cam_ok = _CAMERA_OK
                fps = _FPS
                now = time.monotonic()
                gesture = _read_gesture_unlocked(now)
            if frame is None:
                frame = blank.copy()
                msg = "starting camera..." if cam_ok else "camera unavailable"
                cv2.putText(frame, msg, (20, CAMERA_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                if pts:
                    for a, b in _HAND_CONNECTIONS:
                        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                    for x, y in pts:
                        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"gesture: {gesture}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 0), 2)
                cv2.putText(frame, f"fps: {fps:.0f}", (10, CAMERA_HEIGHT - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Gesture Engine - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()
        stop()


atexit.register(stop)


if __name__ == "__main__":
    run_debug()