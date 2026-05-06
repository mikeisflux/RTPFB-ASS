from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from rtpfb.hud import HUDState, draw_hud


def test_hud_state_tick_advances_fps():
    s = HUDState()
    s.tick()
    s.tick()
    assert s.fps_ema >= 0


def test_draw_hud_returns_same_shape():
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    state = HUDState(fps_ema=42.0, voice_state="knn-vc:alice")
    out = draw_hud(frame, state)
    assert out.shape == frame.shape
    # The HUD region has been drawn on (mean differs from a flat 128).
    assert out[12:80, 12:260].mean() != 128
