"""
core/smoother.py  (v2)
────────────────
Two-stage smoothing that's responsive enough to not break letters
but still kills camera jitter.

Key change from v1:
  alpha raised to 0.72 (was 0.55) — follows the finger much more
  closely so strokes don't feel floaty or lag behind your motion.
  MAX_JUMP raised to 180px — lets natural fast strokes through.
"""

import math
from collections import deque


class Smoother:
    MAX_JUMP = 180    # larger = allows faster natural motion

    def __init__(self, alpha: float = 0.72, history: int = 5):
        self.alpha    = alpha
        self._sx      = None
        self._sy      = None
        self._history = deque(maxlen=history)

    def smooth(self, raw_x: int, raw_y: int) -> tuple:
        if self._sx is None:
            self._sx, self._sy = float(raw_x), float(raw_y)
            self._history.append((self._sx, self._sy))
            return raw_x, raw_y

        dx   = raw_x - self._sx
        dy   = raw_y - self._sy
        dist = math.hypot(dx, dy)

        if dist > self.MAX_JUMP:
            scale = self.MAX_JUMP / dist
            raw_x = int(self._sx + dx * scale)
            raw_y = int(self._sy + dy * scale)

        self._sx = self.alpha * raw_x + (1 - self.alpha) * self._sx
        self._sy = self.alpha * raw_y + (1 - self.alpha) * self._sy

        self._history.append((self._sx, self._sy))
        return int(self._sx), int(self._sy)

    def velocity(self) -> float:
        if len(self._history) < 2:
            return 0.0
        x0, y0 = self._history[0]
        x1, y1 = self._history[-1]
        return math.hypot(x1 - x0, y1 - y0) / max(len(self._history) - 1, 1)

    def reset(self):
        self._sx = None
        self._sy = None
        self._history.clear()

    def reset_to(self, x: int, y: int):
        self._sx = float(x)
        self._sy = float(y)
        self._history.clear()
        self._history.append((self._sx, self._sy))