"""
core/stroke_engine.py  (v5 — single light blue, slim strokes)
"""

import cv2
import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


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
    size: int = 1

@dataclass
class Stroke:
    points: List[StrokePoint] = field(default_factory=list)
    age: float = 0.0

# ── Tuning — change these to adjust the look ──────────────────────────────────
BASE_WIDTH      = 3       # ← stroke thickness (try 2–8)
INK_DECAY       = 0.97
MAX_GAP         = 10
INTERP_STEPS    = 8
MAX_PARTICLES   = 400
PARTICLE_LIFE   = 0.55
GLITTER_DENSITY = 0.45

# Single light-blue palette — all layers are shades of this one colour
CORE_COLOR  = (255, 230, 160)   # BGR: bright icy blue-white core
BLOOM_COLOR = (220, 160,  60)   # BGR: medium blue bloom
HALO_COLOR  = (120,  60,  10)   # BGR: deep blue outer halo
GLITTER_COLORS = [
    (255, 240, 180),   # near-white ice blue
    (255, 220, 120),   # light blue
    (200, 180,  80),   # softer blue
    (255, 255, 220),   # white-blue
]


def _catmull_rom(p0, p1, p2, p3, t):
    t2, t3 = t*t, t*t*t
    x = 0.5*((2*p1[0])+(-p0[0]+p2[0])*t+(2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2+(-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
    y = 0.5*((2*p1[1])+(-p0[1]+p2[1])*t+(2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2+(-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
    return x, y


def _interpolate_stroke(points, steps=INTERP_STEPS):
    if len(points) < 2:
        return points
    dense = []
    n = len(points)
    for i in range(n - 1):
        i0, i1 = max(i-1, 0), i
        i2, i3 = i+1, min(i+2, n-1)
        p0 = (points[i0].x, points[i0].y)
        p1 = (points[i1].x, points[i1].y)
        p2 = (points[i2].x, points[i2].y)
        p3 = (points[i3].x, points[i3].y)
        avg_speed = (points[i1].speed + points[i2].speed) / 2.0
        for s in range(steps):
            x, y = _catmull_rom(p0, p1, p2, p3, s/steps)
            dense.append(StrokePoint(x, y, avg_speed))
    dense.append(points[-1])
    return dense


class StrokeEngine:
    def __init__(self, width: int, height: int):
        self.W = width
        self.H = height
        self._canvas   = np.zeros((height, width, 3), dtype=np.float32)
        self._glitter  = np.zeros((height, width, 3), dtype=np.float32)
        self._strokes:   List[Stroke]   = []
        self._particles: List[Particle] = []
        self._current:   Optional[Stroke] = None

    def add_point(self, x: int, y: int, dt: float):
        if self._current is None:
            self._current = Stroke()
            self._strokes.append(self._current)

        speed = 0.0
        if self._current.points:
            prev  = self._current.points[-1]
            dist  = math.hypot(x - prev.x, y - prev.y)
            speed = dist / max(dt, 1e-4)
            if dist > MAX_GAP:
                steps = max(2, int(dist / MAX_GAP))
                for k in range(1, steps):
                    t  = k / steps
                    ix = prev.x + (x - prev.x) * t
                    iy = prev.y + (y - prev.y) * t
                    self._current.points.append(StrokePoint(ix, iy, speed))

        self._current.points.append(StrokePoint(float(x), float(y), speed))
        self._spawn_glitter(x, y, speed)

    def lift_pen(self):
        self._current = None

    def clear(self):
        self._canvas[:]  = 0
        self._glitter[:] = 0
        self._strokes.clear()
        self._particles.clear()
        self._current = None

    def render(self, dt: float, glow_mode: bool) -> np.ndarray:
        self._canvas  *= INK_DECAY
        self._glitter *= 0.80

        for stroke in self._strokes:
            self._draw_stroke(stroke)

        self._update_particles(dt)
        self._draw_particles()

        merged = 255.0 - (255.0 - self._canvas) * (255.0 - self._glitter) / 255.0

        for s in self._strokes:
            if s is not self._current:
                s.age += dt
        self._strokes = [s for s in self._strokes
                         if s is self._current or s.age < 45.0]

        return np.clip(merged, 0, 255).astype(np.uint8)

    def _draw_stroke(self, stroke: Stroke):
        pts = stroke.points
        if len(pts) < 2:
            if pts:
                self._dot(int(pts[0].x), int(pts[0].y))
            return

        dense = _interpolate_stroke(pts)

        for i in range(1, len(dense)):
            p0, p1 = dense[i-1], dense[i]
            if abs(p1.x-p0.x) < 0.3 and abs(p1.y-p0.y) < 0.3:
                continue

            width = max(1, BASE_WIDTH)   # fixed slim width
            pt0   = (int(p0.x), int(p0.y))
            pt1   = (int(p1.x), int(p1.y))

            # 3-layer light-blue glow
            cv2.line(self._canvas, pt0, pt1, HALO_COLOR,  width+14, cv2.LINE_AA)
            cv2.line(self._canvas, pt0, pt1, BLOOM_COLOR, width+5,  cv2.LINE_AA)
            cv2.line(self._canvas, pt0, pt1, CORE_COLOR,  width,    cv2.LINE_AA)

            # Round caps so segments join cleanly
            r = max(1, width // 2)
            cv2.circle(self._canvas, pt0, r, CORE_COLOR, -1, cv2.LINE_AA)
            cv2.circle(self._canvas, pt1, r, CORE_COLOR, -1, cv2.LINE_AA)

            # Inline glitter sparkles
            if random.random() < GLITTER_DENSITY:
                gx = int((p0.x + p1.x) / 2) + random.randint(-6, 6)
                gy = int((p0.y + p1.y) / 2) + random.randint(-6, 6)
                self._paint_glitter_dot(gx, gy)

    def _dot(self, x, y):
        cv2.circle(self._canvas, (x,y), BASE_WIDTH+10, HALO_COLOR,  -1, cv2.LINE_AA)
        cv2.circle(self._canvas, (x,y), BASE_WIDTH+4,  BLOOM_COLOR, -1, cv2.LINE_AA)
        cv2.circle(self._canvas, (x,y), BASE_WIDTH,    CORE_COLOR,  -1, cv2.LINE_AA)

    def _paint_glitter_dot(self, x, y):
        if not (2 <= x < self.W-2 and 2 <= y < self.H-2):
            return
        col  = random.choice(GLITTER_COLORS)
        size = random.randint(1, 4)
        # 4-pointed star
        cv2.line(self._glitter, (x-size,y), (x+size,y), col, 1, cv2.LINE_AA)
        cv2.line(self._glitter, (x,y-size), (x,y+size), col, 1, cv2.LINE_AA)
        cv2.circle(self._glitter, (x,y), 1, (255,255,255), -1, cv2.LINE_AA)

    def _spawn_glitter(self, x, y, speed):
        if len(self._particles) >= MAX_PARTICLES:
            return
        count = 3 + min(6, int(speed / 900))
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            mag   = random.uniform(0.2, 2.8)
            self._particles.append(Particle(
                x=float(x) + random.uniform(-5, 5),
                y=float(y) + random.uniform(-5, 5),
                vx=math.cos(angle)*mag, vy=math.sin(angle)*mag,
                life=PARTICLE_LIFE * random.uniform(0.4, 1.0),
                max_life=PARTICLE_LIFE,
                size=random.randint(1, 3),
            ))

    def _update_particles(self, dt):
        alive = []
        for p in self._particles:
            p.x += p.vx * 60 * dt
            p.y += p.vy * 60 * dt
            p.vx *= 0.91
            p.vy *= 0.91
            p.life -= dt
            if p.life > 0:
                alive.append(p)
        self._particles = alive

    def _draw_particles(self):
        for p in self._particles:
            frac = p.life / p.max_life
            col  = tuple(int(c * frac) for c in random.choice(GLITTER_COLORS))
            x, y = int(p.x), int(p.y)
            if not (0 <= x < self.W and 0 <= y < self.H):
                continue
            s = p.size
            cv2.line(self._glitter, (x-s,y), (x+s,y), col, 1, cv2.LINE_AA)
            cv2.line(self._glitter, (x,y-s), (x,y+s), col, 1, cv2.LINE_AA)
