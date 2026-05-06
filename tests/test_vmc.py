from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


def _install_fake_pythonosc(monkeypatch):
    """Patch python-osc with a recording fake so we can assert what would be
    sent over the wire without needing the dep installed in CI."""
    sent: list[tuple[str, list]] = []

    class FakeClient:
        def __init__(self, host, port):
            self.host = host
            self.port = port

        def send_message(self, address, value):
            if not isinstance(value, list):
                value = [value]
            sent.append((address, list(value)))

    fake_module = MagicMock()
    fake_module.udp_client.SimpleUDPClient = FakeClient
    monkeypatch.setitem(sys.modules, "pythonosc", fake_module)
    monkeypatch.setitem(sys.modules, "pythonosc.udp_client", fake_module.udp_client)
    return sent


class FakePose:
    def __init__(self, pose_landmarks=None, face_landmarks=None):
        self.pose_landmarks = pose_landmarks
        self.face_landmarks = face_landmarks
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None


def _make_pose_lms() -> np.ndarray:
    return np.array(
        [[float(i % 10) / 10.0, float(i % 7) / 7.0, float(i % 3) / 3.0] for i in range(33)],
        dtype=np.float32,
    )


def test_vmc_sender_emits_root_and_bones(monkeypatch):
    sent = _install_fake_pythonosc(monkeypatch)
    from rtpfb.vmc import VMCSender

    sender = VMCSender(host="127.0.0.1", port=39539)
    pose = FakePose(pose_landmarks=_make_pose_lms())
    sender.send(pose)

    addresses = [s[0] for s in sent]
    assert "/VMC/Ext/Root/Pos" in addresses
    assert addresses.count("/VMC/Ext/Bone/Pos") >= 10  # we send a bunch of bones
    assert addresses[-1] == "/VMC/Ext/OK"


def test_vmc_sender_skips_when_no_pose(monkeypatch):
    sent = _install_fake_pythonosc(monkeypatch)
    from rtpfb.vmc import VMCSender

    sender = VMCSender()
    sender.send(FakePose(pose_landmarks=None))
    assert sent == []


def test_vmc_sender_emits_blendshapes(monkeypatch):
    sent = _install_fake_pythonosc(monkeypatch)
    from rtpfb.vmc import VMCSender

    sender = VMCSender()
    pose = FakePose(pose_landmarks=_make_pose_lms())
    sender.send(pose, blendshapes={"jawOpen": 0.5, "eyeBlinkLeft": 1.0})

    addresses = [s[0] for s in sent]
    blend_packets = [s for s in sent if s[0] == "/VMC/Ext/Blend/Val"]
    assert len(blend_packets) == 2
    assert any(p[1][0] == "jawOpen" for p in blend_packets)
    assert "/VMC/Ext/Blend/Apply" in addresses


def test_vmc_sender_dependency_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "pythonosc", None)
    monkeypatch.setitem(sys.modules, "pythonosc.udp_client", None)
    # Force ImportError by removing it
    sys.modules.pop("pythonosc", None)
    sys.modules.pop("pythonosc.udp_client", None)

    # Block import path
    import builtins

    real_import = builtins.__import__

    def _block(name, *args, **kwargs):
        if name.startswith("pythonosc"):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block)

    from rtpfb.errors import DependencyMissingError
    from rtpfb.vmc import VMCSender

    with pytest.raises(DependencyMissingError):
        VMCSender()


def test_blendshapes_from_face_landmarks_returns_empty_when_none():
    from rtpfb.vmc import blendshapes_from_face_landmarks

    assert blendshapes_from_face_landmarks(None) == {}


def test_blendshapes_from_face_landmarks_synthesises_keys():
    from rtpfb.vmc import blendshapes_from_face_landmarks

    # Build a fake (478, 3) array with a recognisable mouth opening.
    lm = np.zeros((478, 3), dtype=np.float32)
    # forehead high, chin low
    lm[10] = [0.5, 0.1, 0.0]
    lm[152] = [0.5, 0.9, 0.0]
    # mouth half open
    lm[13] = [0.5, 0.45, 0.0]
    lm[14] = [0.5, 0.55, 0.0]
    # eyes half closed
    lm[159] = [0.4, 0.30, 0.0]
    lm[145] = [0.4, 0.32, 0.0]
    lm[386] = [0.6, 0.30, 0.0]
    lm[374] = [0.6, 0.32, 0.0]

    bs = blendshapes_from_face_landmarks(lm)
    assert set(bs) == {"jawOpen", "eyeBlinkLeft", "eyeBlinkRight"}
    for v in bs.values():
        assert 0.0 <= v <= 1.0
    assert bs["jawOpen"] > 0.1
