from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from rtpfb.stabilize import TemporalStabilizer


def test_passthrough_first_frame():
    s = TemporalStabilizer(blend=0.5)
    frame = np.full((64, 64, 3), 128, dtype=np.uint8)
    out = s.smooth(frame)
    np.testing.assert_array_equal(out, frame)


def test_blends_subsequent_frames():
    s = TemporalStabilizer(blend=0.5)
    f1 = np.zeros((64, 64, 3), dtype=np.uint8)
    f2 = np.full((64, 64, 3), 200, dtype=np.uint8)
    s.smooth(f1)
    out = s.smooth(f2)
    # Static scene: previous frame warps to itself (~0); 50/50 blend pulls f2 down.
    assert out.mean() < f2.mean()
    assert out.mean() > 50


def test_reset_clears_history():
    s = TemporalStabilizer(blend=0.5)
    s.smooth(np.zeros((32, 32, 3), dtype=np.uint8))
    assert s._prev_gray is not None
    s.reset()
    assert s._prev_gray is None
    assert s._prev_out is None
