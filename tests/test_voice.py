from __future__ import annotations

import numpy as np
import pytest

from rtpfb.voice import StreamingVoiceChanger


def _bare_changer(crossfade_samples: int = 4) -> StreamingVoiceChanger:
    """Build a StreamingVoiceChanger without running __init__ (which loads
    heavy backends)."""
    vc = StreamingVoiceChanger.__new__(StreamingVoiceChanger)
    vc.crossfade_samples = crossfade_samples
    vc._tail = np.zeros(crossfade_samples, dtype=np.float32)
    return vc


def test_auto_detect_no_voice_falls_back():
    assert StreamingVoiceChanger._auto_detect_backend("") == "w-okada"


def test_auto_detect_unknown_voice_falls_back():
    # Unknown voice name -> not in library -> w-okada (treated as remote slot id)
    assert StreamingVoiceChanger._auto_detect_backend("definitely-not-registered") == "w-okada"


def test_crossfade_short_chunk_is_passthrough():
    vc = _bare_changer(4)
    chunk = np.array([1, 2, 3], dtype=np.float32)
    out = vc._crossfade(chunk)
    np.testing.assert_array_equal(out, chunk)


def test_crossfade_blends_tail_then_passes_remainder():
    vc = _bare_changer(2)
    vc._tail = np.array([1.0, 1.0], dtype=np.float32)
    chunk = np.array([0.0, 0.0, 5.0, 6.0], dtype=np.float32)
    out = vc._crossfade(chunk)
    # ramp = [0, 1]; head[i] = chunk[i]*ramp[i] + tail[i]*(1-ramp[i])
    np.testing.assert_array_almost_equal(out[:2], [1.0, 0.0])
    np.testing.assert_array_almost_equal(out[2:], [5.0, 6.0])
    np.testing.assert_array_almost_equal(vc._tail, [5.0, 6.0])


def test_crossfade_updates_tail_each_call():
    vc = _bare_changer(2)
    vc._tail = np.zeros(2, dtype=np.float32)
    out1 = vc._crossfade(np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_almost_equal(vc._tail, [2.0, 3.0])
    out2 = vc._crossfade(np.array([0.0, 0.0, 9.0, 9.0], dtype=np.float32))
    np.testing.assert_array_almost_equal(vc._tail, [9.0, 9.0])
    # No assertions on out1/out2 beyond confirming the call doesn't blow up.
    assert out1.shape == (4,)
    assert out2.shape == (4,)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        StreamingVoiceChanger(backend="bogus")
