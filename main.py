"""
AIR WRITING — main.py (v3, Windows-compatible)

Controls:
  ✦  Index finger UP  →  Draw
  ✦  Fist             →  Lift pen
  ✦  Open palm        →  Clear canvas
  ✦  Peace sign       →  Toggle glow mode
  ✦  Q key            →  Quit
"""

import cv2
import sys
import time
import numpy as np

from core.hand_tracker  import HandTracker
from core.stroke_engine import StrokeEngine
from core.smoother      import Smoother
from core.gesture       import GestureDetector
from core.composer      import FrameComposer


def open_camera():
    """
    Try multiple camera backends until one works.
    On Windows, DSHOW (DirectShow) is far more reliable than the default.
    """
    backends = [
        (0, cv2.CAP_DSHOW,  "index 0 + DirectShow"),
        (0, cv2.CAP_MSMF,   "index 0 + MSMF"),
        (0, cv2.CAP_ANY,    "index 0 + AUTO"),
        (1, cv2.CAP_DSHOW,  "index 1 + DirectShow"),
        (1, cv2.CAP_ANY,    "index 1 + AUTO"),
    ]

    for idx, backend, label in backends:
        print(f"[INFO] Trying camera {label} ...")
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue

        # Give Windows a moment to initialise the device
        time.sleep(0.3)
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            print(f"[INFO] Camera opened: {label}")
            return cap
        cap.release()

    print("[ERROR] No working camera found.")
    print("        • Make sure your webcam is plugged in")
    print("        • Close other apps using the camera (Teams, Zoom, etc.)")
    print("        • Try running as Administrator")
    sys.exit(1)


def main():
    # ── Camera ────────────────────────────────────────────────────────────────
    cap = open_camera()

    # Lower to 640×480 for reliability & performance on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    # Read actual resolution (may differ from requested)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Resolution: {W}×{H}")

    # ── Systems ───────────────────────────────────────────────────────────────
    tracker  = HandTracker()
    smoother = Smoother(alpha=0.72, history=5)
    gestures = GestureDetector()
    engine   = StrokeEngine(W, H)
    composer = FrameComposer(W, H)

    # ── Window ────────────────────────────────────────────────────────────────
    WINDOW = "AIR WRITING"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # ── State ─────────────────────────────────────────────────────────────────
    drawing     = False
    glow_mode   = True
    prev_time   = time.time()
    fps_display = 30.0
    intro_start = time.time()
    drop_count  = 0
    MAX_DROPS   = 10   # restart camera if this many consecutive drops

    print("[INFO] Ready — raise your index finger and write!")

    while True:
        ret, frame = cap.read()

        # ── Frame-drop recovery ───────────────────────────────────────────────
        if not ret or frame is None or frame.size == 0:
            drop_count += 1
            if drop_count >= MAX_DROPS:
                print("[WARN] Too many dropped frames — reopening camera...")
                cap.release()
                time.sleep(1.0)
                cap = open_camera()
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
                drop_count = 0
            continue
        drop_count = 0

        frame = cv2.flip(frame, 1)   # mirror for natural writing feel

        # ── Timing ───────────────────────────────────────────────────────────
        now  = time.time()
        dt   = max(now - prev_time, 1e-4)
        prev_time = now
        fps_display = 0.9 * fps_display + 0.1 / dt

        # ── Hand tracking ─────────────────────────────────────────────────────
        landmarks = tracker.process(frame)

        if landmarks:
            raw_x = int(landmarks[8].x * W)
            raw_y = int(landmarks[8].y * H)
            sx, sy = smoother.smooth(raw_x, raw_y)

            gesture = gestures.detect(landmarks, W, H)

            if gesture == "CLEAR":
                engine.clear()
                drawing = False
                smoother.reset()
            elif gesture == "TOGGLE_GLOW":
                glow_mode = not glow_mode
            elif gesture == "DRAW":
                if not drawing:
                    smoother.reset_to(sx, sy)
                    drawing = True
                engine.add_point(sx, sy, dt)
            else:
                if drawing:
                    engine.lift_pen()
                drawing = False
        else:
            if drawing:
                engine.lift_pen()
            drawing = False
            smoother.reset()

        # ── Render ────────────────────────────────────────────────────────────
        ink_layer = engine.render(dt, glow_mode)
        final     = composer.compose(frame, ink_layer, fps_display,
                                     drawing, glow_mode)

        # Intro fade-in
        elapsed = now - intro_start
        if elapsed < 3.5:
            alpha = max(0.0, 1.0 - elapsed / 2.5)
            composer.draw_intro(final, alpha)

        cv2.imshow(WINDOW, final)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            engine.clear()
        elif key == ord('g'):
            glow_mode = not glow_mode

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done. ✦")


if __name__ == "__main__":
    main()