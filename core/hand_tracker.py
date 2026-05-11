"""
core/hand_tracker.py
────────────────────
Wraps MediaPipe Hands into a clean, single-method interface.
Supports both the old mp.solutions API and the new Tasks API
introduced in mediapipe 0.10.30+.

Landmark index cheat-sheet (the ones we use):
  4  = thumb tip
  8  = index fingertip   ← primary drawing point
  12 = middle fingertip
  16 = ring fingertip
  20 = pinky fingertip
  6  = index PIP (mid-knuckle)   ← used for raised-finger test
"""

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions
import urllib.request
import os
import numpy as np


# Path to the downloaded model file
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def _ensure_model():
    """Download the hand landmark model file if not already present."""
    if not os.path.exists(_MODEL_PATH):
        print("[INFO] Downloading MediaPipe hand model (~8 MB) — one-time only...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[INFO] Model downloaded.")


class _LandmarkWrapper:
    """
    Mimics the old mp.solutions landmark object so the rest of the
    code (gesture.py etc.) can keep using  lm[8].x / lm[8].y  syntax.
    """
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class HandTracker:
    """Detects one hand and returns its landmarks every frame."""

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.75,
        tracking_confidence:  float = 0.65,
    ):
        _ensure_model()

        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            min_hand_presence_confidence=detection_confidence,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)

    def process(self, bgr_frame):
        """
        Feed one BGR webcam frame; returns a list of 21 landmark
        wrapper objects (with .x/.y in [0,1]) or None.
        """
        # Convert BGR numpy array → MediaPipe Image (RGB internally)
        rgb = bgr_frame[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._detector.detect(mp_image)

        if result.hand_landmarks:
            # Wrap into simple objects with .x/.y/.z attributes
            raw = result.hand_landmarks[0]   # first hand
            return [_LandmarkWrapper(lm.x, lm.y, lm.z) for lm in raw]
        return None

    def close(self):
        """Release MediaPipe resources explicitly if needed."""
        self._detector.close()