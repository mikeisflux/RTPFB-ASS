from __future__ import annotations

from unittest.mock import patch

import pytest

from rtpfb.voice_library import (
    METHOD_KNNVC,
    KNNVCEmbedder,
    VoiceLibrary,
    VoiceMeta,
)


@pytest.fixture
def fake_samples(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "a.wav").write_bytes(b"\x00\x00")
    (samples / "b.wav").write_bytes(b"\x00\x00")
    return samples


def _stub_embed(self, sample_paths, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "features.pt").write_bytes(b"")


def test_add_creates_layout(fake_samples, tmp_path):
    lib = VoiceLibrary(root=tmp_path / "voices")
    with patch.object(KNNVCEmbedder, "embed", _stub_embed):
        meta = lib.add("alice", fake_samples, method=METHOD_KNNVC, accent_tag="en-US")

    voice_dir = lib.voice_dir("alice")
    assert (voice_dir / "samples" / "a.wav").exists()
    assert (voice_dir / "knnvc" / "features.pt").exists()
    assert (voice_dir / "meta.json").exists()
    assert meta.name == "alice"
    assert meta.method == METHOD_KNNVC
    assert meta.accent_tag == "en-US"


def test_list_and_get(fake_samples, tmp_path):
    lib = VoiceLibrary(root=tmp_path / "voices")
    with patch.object(KNNVCEmbedder, "embed", _stub_embed):
        lib.add("alice", fake_samples)
        lib.add("bob", fake_samples)

    names = sorted(v.name for v in lib.list_voices())
    assert names == ["alice", "bob"]
    assert lib.get("alice").name == "alice"


def test_get_missing_raises(tmp_path):
    lib = VoiceLibrary(root=tmp_path / "voices")
    with pytest.raises(KeyError):
        lib.get("nope")


def test_remove(fake_samples, tmp_path):
    lib = VoiceLibrary(root=tmp_path / "voices")
    with patch.object(KNNVCEmbedder, "embed", _stub_embed):
        lib.add("alice", fake_samples)
    lib.remove("alice")
    assert lib.list_voices() == []


def test_existing_without_overwrite_raises(fake_samples, tmp_path):
    lib = VoiceLibrary(root=tmp_path / "voices")
    with patch.object(KNNVCEmbedder, "embed", _stub_embed):
        lib.add("alice", fake_samples)
        with pytest.raises(FileExistsError):
            lib.add("alice", fake_samples)
        lib.add("alice", fake_samples, overwrite=True)


def test_unknown_method_rejected(fake_samples, tmp_path):
    lib = VoiceLibrary(root=tmp_path / "voices")
    with pytest.raises(ValueError):
        lib.add("alice", fake_samples, method="bogus")


def test_empty_samples_rejected(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    lib = VoiceLibrary(root=tmp_path / "voices")
    with pytest.raises(ValueError):
        lib.add("alice", empty)


def test_meta_roundtrip():
    m = VoiceMeta(
        name="x",
        method=METHOD_KNNVC,
        sample_rate=16000,
        samples_seconds=12.5,
        created_at=123.0,
        accent_tag="en-US",
        notes="hello",
    )
    assert VoiceMeta.from_json(m.to_json()) == m


def test_artifact_path(fake_samples, tmp_path):
    lib = VoiceLibrary(root=tmp_path / "voices")
    with patch.object(KNNVCEmbedder, "embed", _stub_embed):
        lib.add("alice", fake_samples, method=METHOD_KNNVC)
    assert lib.artifact_path("alice").name == "features.pt"
