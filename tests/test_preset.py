from __future__ import annotations

import pytest

from rtpfb.errors import ConfigurationError
from rtpfb.preset import config_from_preset, load_preset


def test_load_yaml(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("target_face_path: /tmp/face.png\nfps: 60\n")
    assert load_preset(p) == {"target_face_path": "/tmp/face.png", "fps": 60}


def test_load_empty_yaml_returns_dict(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert load_preset(p) == {}


def test_load_non_mapping_rejected(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ConfigurationError):
        load_preset(p)


def test_config_from_preset(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text(
        """
target_face_path: /tmp/face.png
enable_voice: true
voice_model: alice
fps: 60
""".strip()
    )
    cfg = config_from_preset(p)
    assert cfg.target_face_path == "/tmp/face.png"
    assert cfg.enable_voice is True
    assert cfg.voice_model == "alice"
    assert cfg.fps == 60


def test_overrides_win_over_preset(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("target_face_path: /tmp/face.png\nfps: 30\n")
    cfg = config_from_preset(p, overrides={"fps": 60, "camera_index": 2})
    assert cfg.fps == 60
    assert cfg.camera_index == 2


def test_none_overrides_dropped(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("target_face_path: /tmp/face.png\nfps: 30\n")
    cfg = config_from_preset(p, overrides={"fps": None, "camera_index": 1})
    assert cfg.fps == 30
    assert cfg.camera_index == 1


def test_unknown_keys_rejected(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("target_face_path: /tmp/face.png\nbogus: 1\n")
    with pytest.raises(ConfigurationError):
        config_from_preset(p)


def test_missing_target_rejected_for_faceswap(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("mode: faceswap\nfps: 30\n")
    with pytest.raises(ConfigurationError):
        config_from_preset(p)


def test_missing_target_rejected_for_body_render(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("enable_body: true\nfps: 30\n")
    with pytest.raises(ConfigurationError):
        config_from_preset(p)


def test_missing_target_ok_for_voice_only(tmp_path):
    # Voice-only / mocap-only presets don't load a target face, so the
    # target_face_path requirement should not apply to them.
    p = tmp_path / "p.yaml"
    p.write_text("enable_voice: true\nfps: 30\n")
    cfg = config_from_preset(p)
    assert cfg.enable_voice is True
    assert cfg.target_face_path == ""


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_preset(tmp_path / "missing.yaml")
