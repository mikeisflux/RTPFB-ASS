from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


def _install_fake_pythonosc(monkeypatch):
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
    def __init__(
        self,
        pose_landmarks=None,
        face_landmarks=None,
        left_hand_landmarks=None,
        right_hand_landmarks=None,
    ):
        self.pose_landmarks = pose_landmarks
        self.face_landmarks = face_landmarks
        self.left_hand_landmarks = left_hand_landmarks
        self.right_hand_landmarks = right_hand_landmarks


def _make_pose_lms() -> np.ndarray:
    """Synthesise a plausible 33-pose landmark array (deterministic)."""
    lms = np.zeros((33, 3), dtype=np.float32)
    # Spine column down the middle.
    lms[0] = [0.50, 0.20, 0.00]   # nose
    lms[7] = [0.45, 0.22, 0.00]   # left ear
    lms[8] = [0.55, 0.22, 0.00]   # right ear
    lms[11] = [0.40, 0.35, 0.00]  # left shoulder
    lms[12] = [0.60, 0.35, 0.00]  # right shoulder
    lms[13] = [0.30, 0.50, 0.00]  # left elbow
    lms[14] = [0.70, 0.50, 0.00]  # right elbow
    lms[15] = [0.25, 0.65, 0.00]  # left wrist
    lms[16] = [0.75, 0.65, 0.00]  # right wrist
    lms[23] = [0.45, 0.60, 0.00]  # left hip
    lms[24] = [0.55, 0.60, 0.00]  # right hip
    lms[25] = [0.45, 0.75, 0.00]  # left knee
    lms[26] = [0.55, 0.75, 0.00]  # right knee
    lms[27] = [0.45, 0.90, 0.00]  # left ankle
    lms[28] = [0.55, 0.90, 0.00]  # right ankle
    lms[31] = [0.45, 0.95, 0.05]  # left foot index
    lms[32] = [0.55, 0.95, 0.05]  # right foot index
    return lms


def _make_hand_lms() -> np.ndarray:
    """Synthesise a 21-landmark hand."""
    lms = np.zeros((21, 3), dtype=np.float32)
    for i in range(21):
        lms[i] = [0.5 + 0.01 * i, 0.5 + 0.01 * i, 0.0]
    return lms


def test_emits_root_and_bones_with_rotations(monkeypatch):
    sent = _install_fake_pythonosc(monkeypatch)
    from rtpfb.vmc import VMCSender

    sender = VMCSender(host="127.0.0.1", port=39539)
    sender.send(FakePose(pose_landmarks=_make_pose_lms()))

    addresses = [s[0] for s in sent]
    bone_packets = [s for s in sent if s[0] == "/VMC/Ext/Bone/Pos"]

    assert "/VMC/Ext/Root/Pos" in addresses
    # 25-ish body bones come out of the body chain.
    assert len(bone_packets) >= 18
    # Each bone packet is [name, x, y, z, qx, qy, qz, qw] — 8 elements.
    for _, args in bone_packets:
        assert len(args) == 8
        assert isinstance(args[0], str)

    # End-of-frame marker
    assert addresses[-1] == "/VMC/Ext/OK"


def test_skips_when_no_pose(monkeypatch):
    sent = _install_fake_pythonosc(monkeypatch)
    from rtpfb.vmc import VMCSender

    sender = VMCSender()
    sender.send(FakePose(pose_landmarks=None))
    assert sent == []


def test_emits_finger_bones_when_hands_present(monkeypatch):
    sent = _install_fake_pythonosc(monkeypatch)
    from rtpfb.vmc import VMCSender

    sender = VMCSender()
    sender.send(
        FakePose(
            pose_landmarks=_make_pose_lms(),
            left_hand_landmarks=_make_hand_lms(),
            right_hand_landmarks=_make_hand_lms(),
        )
    )

    bone_names = [s[1][0] for s in sent if s[0] == "/VMC/Ext/Bone/Pos"]
    finger_names = {n for n in bone_names if "Thumb" in n or "Index" in n or "Middle" in n or "Ring" in n or "Little" in n}
    # 5 fingers × 3 joints × 2 hands = 30 finger bones
    assert len(finger_names) == 30
    assert "LeftThumbProximal" in finger_names
    assert "RightLittleDistal" in finger_names


def test_emits_blendshapes(monkeypatch):
    sent = _install_fake_pythonosc(monkeypatch)
    from rtpfb.vmc import VMCSender

    sender = VMCSender()
    sender.send(
        FakePose(pose_landmarks=_make_pose_lms()),
        blendshapes={"jawOpen": 0.5, "eyeBlinkLeft": 1.0},
    )

    addresses = [s[0] for s in sent]
    blend_packets = [s for s in sent if s[0] == "/VMC/Ext/Blend/Val"]
    assert len(blend_packets) == 2
    assert "/VMC/Ext/Blend/Apply" in addresses


def test_dependency_missing(monkeypatch):
    sys.modules.pop("pythonosc", None)
    sys.modules.pop("pythonosc.udp_client", None)

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

    lm = np.zeros((478, 3), dtype=np.float32)
    lm[10] = [0.5, 0.1, 0.0]
    lm[152] = [0.5, 0.9, 0.0]
    lm[13] = [0.5, 0.45, 0.0]
    lm[14] = [0.5, 0.55, 0.0]
    lm[159] = [0.4, 0.30, 0.0]
    lm[145] = [0.4, 0.32, 0.0]
    lm[386] = [0.6, 0.30, 0.0]
    lm[374] = [0.6, 0.32, 0.0]

    bs = blendshapes_from_face_landmarks(lm)
    assert set(bs) == {"jawOpen", "eyeBlinkLeft", "eyeBlinkRight"}
    for v in bs.values():
        assert 0.0 <= v <= 1.0
    assert bs["jawOpen"] > 0.1


def test_compute_body_transforms_produces_expected_bones():
    from rtpfb.vmc import _compute_body_transforms

    transforms = _compute_body_transforms(FakePose(pose_landmarks=_make_pose_lms()))
    expected = {
        "Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
        "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
        "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
        "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
        "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
    }
    assert expected.issubset(set(transforms))
    # Each value is (position[3], quat[4]).
    for pos, rot in transforms.values():
        assert pos.shape == (3,)
        assert rot.shape == (4,)


def test_compute_finger_transforms_15_bones():
    from rtpfb.vmc import _compute_finger_transforms

    out = _compute_finger_transforms("Left", _make_hand_lms())
    assert len(out) == 15
    for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
        for segment in ("Proximal", "Intermediate", "Distal"):
            assert f"Left{finger}{segment}" in out


def test_compute_finger_transforms_empty_when_no_hand():
    from rtpfb.vmc import _compute_finger_transforms

    assert _compute_finger_transforms("Left", None) == {}
    assert _compute_finger_transforms("Left", np.zeros((10, 3), dtype=np.float32)) == {}
