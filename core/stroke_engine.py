"""
core/stroke_engine.py
(Jalebi Air Canvas — ultra smooth version)
"""

import cv2  # type: ignore
import math
import numpy as np  # type: ignore
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StrokePoint:
    x: float
    y: float
    speed: float = 0.0


@dataclass
class Stroke:
    points: List[StrokePoint] = field(default_factory=list)
    age: float = 0.0


# ── Jalebi Visual Settings ────────────────────────────

INK_DECAY = 0.998

# Warm syrup glow 🍯
CORE_COLOR = (0, 180, 255)
BLOOM_COLOR = (0, 120, 220)
HALO_COLOR = (0, 70, 140)


class StrokeEngine:

    def __init__(self, width: int, height: int):

        self.W = width
        self.H = height

        self._canvas = np.zeros(
            (height, width, 3),
            dtype=np.float32
        )

        self._strokes: List[Stroke] = []
        self._current: Optional[Stroke] = None

        # Smooth tracking
        self._smooth_x = None
        self._smooth_y = None

    def add_point(self, x: int, y: int, dt: float):

        # ── Finger smoothing ─────────────────────
        alpha = 0.18

        if self._smooth_x is None:
            self._smooth_x = x
            self._smooth_y = y

        else:
            self._smooth_x = (
                alpha * x +
                (1 - alpha) * self._smooth_x
            )

            self._smooth_y = (
                alpha * y +
                (1 - alpha) * self._smooth_y
            )

        x = int(self._smooth_x)
        y = int(self._smooth_y)

        # ── Create stroke if needed ─────────────
        if self._current is None:
            self._current = Stroke()
            self._strokes.append(self._current)

        speed = 0.0

        if self._current.points:

            prev = self._current.points[-1]

            dist = math.hypot(
                x - prev.x,
                y - prev.y
            )

            speed = dist / max(dt, 1e-4)

        self._current.points.append(
            StrokePoint(
                float(x),
                float(y),
                speed
            )
        )

    def lift_pen(self):
        self._current = None

    def clear(self):

        self._canvas[:] = 0

        self._strokes.clear()

        self._current = None

    def render(self, dt: float, glow_mode: bool) -> np.ndarray:

        # Smooth fading trail
        self._canvas *= INK_DECAY

        # Draw all strokes
        for stroke in self._strokes:
            self._draw_stroke(stroke)

        # Age strokes
        for s in self._strokes:
            if s is not self._current:
                s.age += dt

        self._strokes = [
            s for s in self._strokes
            if s is self._current or s.age < 45.0
        ]

        return np.clip(
            self._canvas,
            0,
            255
        ).astype(np.uint8)

    def _draw_stroke(self, stroke: Stroke):

        pts = stroke.points

        if len(pts) < 2:
            return

        for i in range(1, len(pts)):

            p0 = pts[i - 1]
            p1 = pts[i]

            pt0 = (
                int(p0.x),
                int(p0.y)
            )

            pt1 = (
                int(p1.x),
                int(p1.y)
            )

            # ── Outer soft glow ─────────────────
            cv2.line(
                self._canvas,
                pt0,
                pt1,
                HALO_COLOR,
                5,
                cv2.LINE_AA
            )

            # ── Mid glow ────────────────────────
            cv2.line(
                self._canvas,
                pt0,
                pt1,
                BLOOM_COLOR,
                3,
                cv2.LINE_AA
            )

            # ── Core syrup line ─────────────────
            cv2.line(
                self._canvas,
                pt0,
                pt1,
                CORE_COLOR,
                2,
                cv2.LINE_AA
            )