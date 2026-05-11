"""
core/composer.py
────────────────
Final compositing stage — takes the raw webcam frame and the ink
canvas and merges them into the cinematic output the viewer sees.

Pipeline:
  ① Darken webcam feed dramatically (the "shadow world" behind the writing)
  ② Screen-blend the ink layer on top
     (Screen = 1 - (1-A)*(1-B) — bright ink pops even on dark backgrounds)
  ③ Add subtle vignette to focus attention on the centre
  ④ Optionally add a minimal HUD (FPS + mode indicator only)
  ⑤ Optionally draw intro title card

Screen blend is perfect here because:
  • Black ink (0,0,0) + any background = just the background (invisible)
  • White ink (255,255,255) + any background = white (always visible)
  • Coloured glow blends naturally without looking "pasted on"
"""

import cv2
import math
import numpy as np


class FrameComposer:
    def __init__(self, width: int, height: int):
        self.W = width
        self.H = height

        # Pre-build vignette mask (only computed once — expensive otherwise)
        self._vignette = self._build_vignette(width, height, strength=0.65)

        # Pre-build intro title card (rendered once, reused)
        self._intro_card = self._build_intro(width, height)

    # ── Public API ────────────────────────────────────────────────────────────

    def compose(
        self,
        webcam_frame: np.ndarray,
        ink_layer: np.ndarray,
        fps: float,
        drawing: bool,
        glow_mode: bool,
    ) -> np.ndarray:
        """
        Merge webcam + ink into the final output frame.

        Parameters
        ----------
        webcam_frame : Raw BGR uint8 from the camera (already flipped).
        ink_layer    : BGR uint8 from StrokeEngine — black background with
                       bright strokes drawn on it.
        fps          : Current FPS for HUD display.
        drawing      : Whether we are currently in draw mode.
        glow_mode    : For colouring the HUD dot.

        Returns
        -------
        Composited BGR uint8 frame ready for cv2.imshow.
        """
        # ── Step 1: Dramatically darken the webcam feed ───────────────────────
        # We keep it visible but very dim so the background is present
        # (feels like writing on glass in a dark room).
        dark = (webcam_frame.astype(np.float32) * 0.18).clip(0, 255)

        # ── Step 2: Screen blend dark webcam + ink layer ──────────────────────
        ink_f   = ink_layer.astype(np.float32)
        dark_f  = dark

        # Screen formula: out = 255 - (255-A)*(255-B)/255
        blended = 255.0 - (255.0 - dark_f) * (255.0 - ink_f) / 255.0
        blended = blended.clip(0, 255)

        # ── Step 3: Vignette (darken edges) ───────────────────────────────────
        blended *= self._vignette

        result = blended.clip(0, 255).astype(np.uint8)

        # ── Step 4: HUD ───────────────────────────────────────────────────────
        self._draw_hud(result, fps, drawing, glow_mode)

        return result

    def draw_intro(self, frame: np.ndarray, alpha: float):
        """Overlay the intro title with a given opacity (1.0→0.0 fade)."""
        if alpha <= 0:
            return
        overlay = self._intro_card.astype(np.float32) * alpha
        frame[:] = np.clip(
            frame.astype(np.float32) + overlay, 0, 255
        ).astype(np.uint8)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_vignette(w: int, h: int, strength: float = 0.6) -> np.ndarray:
        """
        Create a radial gradient mask: bright centre, dark edges.
        strength=1.0 makes corners fully black; 0=no vignette.
        """
        cx, cy = w / 2, h / 2
        Y, X   = np.ogrid[:h, :w]
        # Normalised distance from centre (0 at centre, 1 at corners)
        dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
        dist = dist.clip(0, 1)
        # Smooth falloff: 1 at centre, (1-strength) at corners
        mask = 1.0 - strength * (dist ** 1.8)
        mask = mask.clip(0, 1).astype(np.float32)
        # Expand to 3 channels
        return mask[:, :, np.newaxis]

    @staticmethod
    def _build_intro(w: int, h: int) -> np.ndarray:
        """
        Render the intro title card on a transparent (black) base.
        This gets alpha-blended during the first few seconds.
        """
        card = np.zeros((h, w, 3), dtype=np.float32)

        # Main title
        title      = "AIR  WRITING"
        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 2.2
        thickness  = 2
        (tw, th), _ = cv2.getTextSize(title, font, font_scale, thickness)
        tx = (w - tw) // 2
        ty = h // 2 - 20

        # Glow layers for the intro title
        for radius, brightness in [(30, 20), (15, 60), (5, 140), (0, 255)]:
            col = (brightness, brightness, brightness)
            if radius > 0:
                # Draw and blur for glow
                tmp = np.zeros_like(card, dtype=np.uint8)
                cv2.putText(tmp, title, (tx, ty), font, font_scale,
                            (255, 255, 255), thickness + 1, cv2.LINE_AA)
                blurred = cv2.GaussianBlur(tmp.astype(np.float32),
                                           (radius * 2 + 1, radius * 2 + 1), 0)
                card += blurred * (brightness / 255.0)
            else:
                cv2.putText(card, title, (tx, ty), font, font_scale,
                            (float(brightness),) * 3, thickness, cv2.LINE_AA)

        # Subtitle
        sub        = "raise index finger to write"
        sub_scale  = 0.65
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_PLAIN, sub_scale, 1)
        cv2.putText(card, sub,
                    ((w - sw * 3) // 2, ty + 55),
                    cv2.FONT_HERSHEY_PLAIN, sub_scale,
                    (120.0, 120.0, 120.0), 1, cv2.LINE_AA)

        return card.clip(0, 255)

    def _draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        drawing: bool,
        glow_mode: bool,
    ):
        """
        Minimal floating HUD — just FPS and a tiny status indicator.
        Nothing that looks like a UI toolbar.
        """
        # FPS counter — top right, very dim
        fps_text = f"{fps:.0f}"
        cv2.putText(frame, fps_text,
                    (self.W - 55, 28),
                    cv2.FONT_HERSHEY_PLAIN, 1.1,
                    (45, 45, 45), 1, cv2.LINE_AA)

        # Tiny status dot — bottom right
        dot_x, dot_y = self.W - 22, self.H - 22
        if drawing:
            if glow_mode:
                cv2.circle(frame, (dot_x, dot_y), 6, (40, 120, 220), -1, cv2.LINE_AA)
                cv2.circle(frame, (dot_x, dot_y), 4, (120, 200, 255),-1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (dot_x, dot_y), 6, (180, 180, 180),-1, cv2.LINE_AA)
                cv2.circle(frame, (dot_x, dot_y), 3, (255, 255, 255),-1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (dot_x, dot_y), 4, (30, 30, 30), -1, cv2.LINE_AA)
