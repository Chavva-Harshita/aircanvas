"""
core/stroke_engine.py  (v2 — smooth continuous strokes)
─────────────────────
Key improvements over v1:
  • Catmull-Rom spline interpolation between points → no broken segments
  • Gap filling: if finger jumps >MAX_GAP pixels, intermediate points are
    synthesised so the line never breaks mid-letter
  • Thicker base width and gentler speed-taper so fast strokes stay visible
  • Smoother glow layers with rounded line caps
"""

import cv2
import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class StrokePoint:
    x: float
    y: float
    speed: float = 0.0


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float


@dataclass
class Stroke:
    points: List[StrokePoint] = field(default_factory=list)
    age: float = 0.0


# ── Tuning constants ──────────────────────────────────────────────────────────

BASE_WIDTH        = 9      # core stroke thickness in pixels
SPEED_FACTOR      = 0.005  # very gentle taper — keeps fast strokes thick
INK_DECAY         = 0.993  # canvas fade per frame
MAX_GAP           = 15     # pixels — fill gaps larger than this
INTERP_STEPS      = 6      # extra points inserted per segment
PARTICLE_SPEED    = 22     # px/frame threshold to spawn sparks
MAX_PARTICLES     = 180
PARTICLE_LIFE     = 0.4


# ── Helpers ───────────────────────────────────────────────────────────────────

def _catmull_rom(p0, p1, p2, p3, t):
    """Single Catmull-Rom spline evaluation at parameter t ∈ [0,1]."""
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * ((2*p1[0]) +
                (-p0[0] + p2[0]) * t +
                (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
                (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
    y = 0.5 * ((2*p1[1]) +
                (-p0[1] + p2[1]) * t +
                (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
                (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
    return x, y


def _interpolate_stroke(points: List[StrokePoint], steps: int = INTERP_STEPS):
    """
    Expand a list of StrokePoints into a much denser list using
    Catmull-Rom splines.  This guarantees smooth curves even when
    MediaPipe delivers positions at ~15-20 Hz.
    """
    if len(points) < 2:
        return points

    dense = []
    n = len(points)

    for i in range(n - 1):
        # Clamp indices for boundary control points
        i0 = max(i - 1, 0)
        i1 = i
        i2 = i + 1
        i3 = min(i + 2, n - 1)

        p0 = (points[i0].x, points[i0].y)
        p1 = (points[i1].x, points[i1].y)
        p2 = (points[i2].x, points[i2].y)
        p3 = (points[i3].x, points[i3].y)

        avg_speed = (points[i1].speed + points[i2].speed) / 2.0

        for s in range(steps):
            t = s / steps
            x, y = _catmull_rom(p0, p1, p2, p3, t)
            dense.append(StrokePoint(x, y, avg_speed))

    dense.append(points[-1])
    return dense


class StrokeEngine:
    """
    Manages strokes and particles; renders them onto a floating ink canvas.
    """

    def __init__(self, width: int, height: int):
        self.W = width
        self.H = height

        # Persistent float32 canvas — fades slowly each frame
        self._canvas   = np.zeros((height, width, 3), dtype=np.float32)

        self._strokes:   List[Stroke]    = []
        self._particles: List[Particle]  = []
        self._current:   Optional[Stroke] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def add_point(self, x: int, y: int, dt: float):
        """Called every frame while the finger is in DRAW mode."""
        if self._current is None:
            self._current = Stroke()
            self._strokes.append(self._current)

        speed = 0.0
        if self._current.points:
            prev  = self._current.points[-1]
            dist  = math.hypot(x - prev.x, y - prev.y)
            speed = dist / max(dt, 1e-4)

            # ── Gap filling ───────────────────────────────────────────────────
            # If the finger jumped a lot in one frame, insert intermediate
            # points so the line doesn't break.
            if dist > MAX_GAP:
                steps = max(2, int(dist / MAX_GAP))
                for k in range(1, steps):
                    t  = k / steps
                    ix = prev.x + (x - prev.x) * t
                    iy = prev.y + (y - prev.y) * t
                    self._current.points.append(StrokePoint(ix, iy, speed))

        self._current.points.append(StrokePoint(float(x), float(y), speed))

        # Particle sparks on fast movement
        if speed > PARTICLE_SPEED * 60 and len(self._particles) < MAX_PARTICLES:
            self._spawn_particles(x, y, speed)

    def lift_pen(self):
        self._current = None

    def clear(self):
        self._canvas[:]  = 0
        self._strokes.clear()
        self._particles.clear()
        self._current = None

    def render(self, dt: float, glow_mode: bool) -> np.ndarray:
        """Render all strokes + particles; return uint8 BGR image."""

        # 1. Fade existing ink
        self._canvas *= INK_DECAY

        # 2. Draw strokes
        for stroke in self._strokes:
            self._draw_stroke(stroke, glow_mode)

        # 3. Particles
        self._update_particles(dt)
        self._draw_particles(glow_mode)

        # 4. Age & prune old strokes
        for stroke in self._strokes:
            if stroke is not self._current:
                stroke.age += dt
        self._strokes = [s for s in self._strokes
                         if s is self._current or s.age < 60.0]

        return np.clip(self._canvas, 0, 255).astype(np.uint8)

    # ── Private rendering ─────────────────────────────────────────────────────

    def _draw_stroke(self, stroke: Stroke, glow_mode: bool):
        pts = stroke.points
        if len(pts) < 2:
            if pts:
                self._dot(int(pts[0].x), int(pts[0].y), glow_mode)
            return

        # Interpolate to smooth curve
        dense = _interpolate_stroke(pts)

        for i in range(1, len(dense)):
            p0, p1 = dense[i - 1], dense[i]

            dx = p1.x - p0.x
            dy = p1.y - p0.y
            if abs(dx) < 0.5 and abs(dy) < 0.5:
                continue

            avg_speed = (p0.speed + p1.speed) / 2.0
            width = max(2, int(BASE_WIDTH / (1 + avg_speed * SPEED_FACTOR)))

            # Brightness: slight dim on very fast strokes
            alpha = max(0.55, 1.0 - avg_speed * 0.00005)

            pt0 = (int(p0.x), int(p0.y))
            pt1 = (int(p1.x), int(p1.y))

            if glow_mode:
                # Warm core, electric blue halo
                halo_col  = (int(15  * alpha), int(8   * alpha), int(4   * alpha))
                bloom_col = (int(160 * alpha), int(90  * alpha), int(15  * alpha))
                core_col  = (int(255 * alpha), int(235 * alpha), int(190 * alpha))
            else:
                d = int(255 * alpha)
                halo_col  = (int(d * 0.04),) * 3
                bloom_col = (int(d * 0.28),) * 3
                core_col  = (d, d, d)

            # Layer 1: wide soft halo
            cv2.line(self._canvas, pt0, pt1, halo_col,  width + 16, cv2.LINE_AA)
            # Layer 2: bloom ring
            cv2.line(self._canvas, pt0, pt1, bloom_col, width + 6,  cv2.LINE_AA)
            # Layer 3: crisp bright core
            cv2.line(self._canvas, pt0, pt1, core_col,  width,      cv2.LINE_AA)

            # Round caps — circles at each endpoint close segment gaps
            cv2.circle(self._canvas, pt0, width // 2, core_col, -1, cv2.LINE_AA)
            cv2.circle(self._canvas, pt1, width // 2, core_col, -1, cv2.LINE_AA)

    def _dot(self, x: int, y: int, glow_mode: bool):
        if glow_mode:
            cv2.circle(self._canvas, (x, y), 12, (15, 8, 4),    -1, cv2.LINE_AA)
            cv2.circle(self._canvas, (x, y),  5, (200, 120, 30),-1, cv2.LINE_AA)
            cv2.circle(self._canvas, (x, y),  2, (255, 235, 190),-1, cv2.LINE_AA)
        else:
            cv2.circle(self._canvas, (x, y), 10, (8, 8, 8),     -1, cv2.LINE_AA)
            cv2.circle(self._canvas, (x, y),  4, (200, 200, 200),-1, cv2.LINE_AA)
            cv2.circle(self._canvas, (x, y),  2, (255, 255, 255),-1, cv2.LINE_AA)

    def _spawn_particles(self, x: int, y: int, speed: float):
        count = min(5, int(speed / 600))
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            mag   = random.uniform(0.5, 2.5)
            self._particles.append(Particle(
                x=float(x), y=float(y),
                vx=math.cos(angle) * mag,
                vy=math.sin(angle) * mag,
                life=PARTICLE_LIFE * random.uniform(0.6, 1.0),
                max_life=PARTICLE_LIFE,
            ))

    def _update_particles(self, dt: float):
        alive = []
        for p in self._particles:
            p.x  += p.vx * 60 * dt
            p.y  += p.vy * 60 * dt
            p.vx *= 0.92
            p.vy *= 0.92
            p.life -= dt
            if p.life > 0:
                alive.append(p)
        self._particles = alive

    def _draw_particles(self, glow_mode: bool):
        for p in self._particles:
            frac = p.life / p.max_life
            brightness = int(255 * frac)
            x, y = int(p.x), int(p.y)
            if 0 <= x < self.W and 0 <= y < self.H:
                color = (brightness // 2, brightness // 3, brightness // 8) \
                        if glow_mode else (brightness,) * 3
                cv2.circle(self._canvas, (x, y), 2, color, -1, cv2.LINE_AA)
