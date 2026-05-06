from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HUDState:
    fps_ema: float = 0.0
    last_t: float = field(default_factory=time.time)
    voice_state: str = "off"
    error_text: str = ""
    audio_buffer_ms: float = 0.0

    def tick(self) -> None:
        now = time.time()
        dt = now - self.last_t
        self.last_t = now
        if dt > 0:
            inst = 1.0 / dt
            self.fps_ema = self.fps_ema * 0.9 + inst * 0.1


def draw_hud(frame_rgb: np.ndarray, state: HUDState) -> np.ndarray:
    """Draw a small telemetry box in the upper-left corner. Mutates ``frame_rgb``."""
    import cv2

    x, y = 12, 12
    pad = 8
    line_h = 18

    lines = [
        f"FPS:    {state.fps_ema:5.1f}",
        f"VOICE:  {state.voice_state}",
    ]
    if state.audio_buffer_ms:
        lines.append(f"AUDIO:  {state.audio_buffer_ms:5.1f} ms")
    if state.error_text:
        lines.append(f"ERR: {state.error_text[:48]}")

    box_w = 260
    box_h = pad * 2 + line_h * len(lines)

    overlay = frame_rgb.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame_rgb, 0.55, 0, frame_rgb)

    for i, line in enumerate(lines):
        cv2.putText(
            frame_rgb,
            line,
            (x + pad, y + pad + line_h * (i + 1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame_rgb
