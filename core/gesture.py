"""
core/gesture.py
───────────────
Classifies hand posture into four named gestures used by the app.

Returned gesture strings:
  "DRAW"        → index finger up, others down  (write!)
  "PAUSE"       → fist / ambiguous              (pen lifted)
  "CLEAR"       → open palm (all 5 fingers up)  (wipe canvas)
  "TOGGLE_GLOW" → peace sign (index + middle)   (switch glow style)

Algorithm: compare each fingertip Y coordinate to its
           corresponding PIP (middle-knuckle) Y coordinate.
           If tip.y < pip.y the finger is *raised* (Y increases downward).
"""


class GestureDetector:
    """Stateless gesture classifier — call detect() every frame."""

    # Landmark indices for tips and their corresponding PIP joints
    # (index, middle, ring, pinky) — thumb handled separately
    _TIPS = [8,  12, 16, 20]
    _PIPS = [6,  10, 14, 18]

    def detect(self, landmarks, frame_w: int, frame_h: int) -> str:
        """
        Parameters
        ----------
        landmarks  : MediaPipe landmark list (21 points)
        frame_w/h  : pixel dimensions (unused here but handy for subclasses)

        Returns
        -------
        One of: "DRAW", "PAUSE", "CLEAR", "TOGGLE_GLOW"
        """
        raised = self._raised_fingers(landmarks)
        # raised is a list of booleans: [index, middle, ring, pinky]

        index_up  = raised[0]
        middle_up = raised[1]
        ring_up   = raised[2]
        pinky_up  = raised[3]

        total_up = sum(raised)

        # ── Clear: open palm ──────────────────────────────────────────────────
        if total_up == 4:
            return "CLEAR"

        # ── Toggle glow: peace / V sign ───────────────────────────────────────
        if index_up and middle_up and not ring_up and not pinky_up:
            return "TOGGLE_GLOW"

        # ── Draw: index only ──────────────────────────────────────────────────
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "DRAW"

        # ── Everything else: pen up ───────────────────────────────────────────
        return "PAUSE"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _raised_fingers(self, lm) -> list:
        """
        Return [bool] for index, middle, ring, pinky.
        A finger is "raised" when its tip is above its PIP joint
        (lower Y value in image coordinates, since Y=0 is top).
        """
        result = []
        for tip_idx, pip_idx in zip(self._TIPS, self._PIPS):
            result.append(lm[tip_idx].y < lm[pip_idx].y)
        return result
