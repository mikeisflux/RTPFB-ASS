from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .config import REPO_ROOT, THIRD_PARTY_DIR

VOICES_DIR = REPO_ROOT / "voices"
SUPPORTED_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}

# Voice cloning backends. Each one defines: how training samples turn into
# whatever artifact we keep on disk, and how the streaming voice changer
# loads that artifact at runtime.
METHOD_KNNVC = "knn-vc"
METHOD_RVC = "rvc"
METHODS = (METHOD_KNNVC, METHOD_RVC)


@dataclass
class VoiceMeta:
    name: str
    method: str
    sample_rate: int
    samples_seconds: float
    created_at: float
    notes: str = ""
    pitch_shift_default: float = 0.0
    accent_tag: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, payload: str) -> "VoiceMeta":
        return cls(**json.loads(payload))


class VoiceLibrary:
    """Filesystem-backed registry of cloned voices.

    Layout::

        voices/
          <name>/
            meta.json
            samples/             # raw training audio, copied in
            knnvc/features.pt    # KNN-VC method only
            rvc/                 # RVC method only
              model.pth
              added.index

    Two backends are supported:

      * ``knn-vc`` — zero-shot cloning. We just cache WavLM features for the
        target speaker; runtime does nearest-neighbour matching. ~1–5 min of
        reference audio is enough. No training run needed.
      * ``rvc`` — full RVC training (much higher quality, much slower).
        Delegates to RVC-WebUI's training scripts as a subprocess.
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = root or VOICES_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def voice_dir(self, name: str) -> Path:
        return self.root / name

    def list_voices(self) -> list[VoiceMeta]:
        voices: list[VoiceMeta] = []
        for entry in sorted(self.root.iterdir()):
            meta_file = entry / "meta.json"
            if entry.is_dir() and meta_file.exists():
                voices.append(VoiceMeta.from_json(meta_file.read_text()))
        return voices

    def get(self, name: str) -> VoiceMeta:
        meta_file = self.voice_dir(name) / "meta.json"
        if not meta_file.exists():
            raise KeyError(f"Voice {name!r} not registered under {self.root}")
        return VoiceMeta.from_json(meta_file.read_text())

    def add(
        self,
        name: str,
        samples_dir: Path,
        method: str = METHOD_KNNVC,
        notes: str = "",
        accent_tag: str = "",
        overwrite: bool = False,
    ) -> VoiceMeta:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}. Pick one of {METHODS}.")
        if not samples_dir.exists():
            raise FileNotFoundError(samples_dir)

        target = self.voice_dir(name)
        if target.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Voice {name!r} already exists. Use --overwrite to replace."
                )
            shutil.rmtree(target)
        target.mkdir(parents=True)

        sample_target = target / "samples"
        sample_target.mkdir()
        copied = self._copy_samples(samples_dir, sample_target)
        if not copied:
            shutil.rmtree(target)
            raise ValueError(
                f"No usable audio in {samples_dir} (looked for: {sorted(SUPPORTED_AUDIO_EXTS)})"
            )

        total_seconds = sum(_audio_duration(p) for p in copied)

        if method == METHOD_KNNVC:
            KNNVCEmbedder().embed(copied, target / "knnvc")
            sr = 16000
        else:
            RVCTrainer().train(copied, target / "rvc", voice_name=name)
            sr = 40000

        meta = VoiceMeta(
            name=name,
            method=method,
            sample_rate=sr,
            samples_seconds=total_seconds,
            created_at=time.time(),
            notes=notes,
            accent_tag=accent_tag,
        )
        (target / "meta.json").write_text(meta.to_json())
        return meta

    def remove(self, name: str) -> None:
        directory = self.voice_dir(name)
        if not directory.exists():
            raise KeyError(name)
        shutil.rmtree(directory)

    def artifact_path(self, name: str) -> Path:
        meta = self.get(name)
        directory = self.voice_dir(name)
        if meta.method == METHOD_KNNVC:
            return directory / "knnvc" / "features.pt"
        return directory / "rvc" / "model.pth"

    def _copy_samples(self, src: Path, dst: Path) -> list[Path]:
        copied: list[Path] = []
        sources: list[Path] = [src] if src.is_file() else sorted(src.rglob("*"))
        for path in sources:
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                dst_path = dst / path.name
                shutil.copy2(path, dst_path)
                copied.append(dst_path)
        return copied


def _audio_duration(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 0.0


class KNNVCEmbedder:
    """Build the matching set for KNN-VC.

    KNN-VC's "training" is just running the reference audio through WavLM and
    keeping the per-frame features. At inference, the source speaker's
    WavLM features are mapped to the closest target features and a
    HiFi-GAN vocoder synthesises the converted waveform.
    """

    def embed(self, sample_paths: list[Path], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        knn_root = THIRD_PARTY_DIR / "knn-vc"
        if not knn_root.exists():
            raise FileNotFoundError(
                f"{knn_root} missing — run scripts/install_third_party.sh"
            )
        if str(knn_root) not in sys.path:
            sys.path.insert(0, str(knn_root))

        import torch

        knn = torch.hub.load("bshall/knn-vc", "knn_vc", trust_repo=True, prematched=True)

        feats = []
        for path in sample_paths:
            feature = knn.get_features(str(path))
            feats.append(feature)
        all_feats = torch.cat(feats, dim=0).cpu()
        torch.save(all_feats, out_dir / "features.pt")


class RVCTrainer:
    """Wrapper around RVC-WebUI's training scripts.

    RVC training has four stages: preprocess → extract f0/features → train
    index → train generator+discriminator. RVC-WebUI exposes these via
    ``infer/modules/train/`` scripts. The exact CLI surface drifts between
    releases, so we surface a clear error here until we pin a tag rather
    than producing a half-trained model.

    Manual fallback: train via the RVC-WebUI GUI, then drop the resulting
    ``.pth`` + ``.index`` into ``voices/<name>/rvc/`` and write a meta.json
    matching ``VoiceMeta``.
    """

    def train(self, sample_paths: list[Path], out_dir: Path, voice_name: str) -> None:  # noqa: ARG002
        out_dir.mkdir(parents=True, exist_ok=True)
        rvc_root = THIRD_PARTY_DIR / "RVC-WebUI"
        if not rvc_root.exists():
            raise FileNotFoundError(
                f"{rvc_root} missing — run scripts/install_third_party.sh"
            )
        raise NotImplementedError(
            "RVC training is not auto-wired. Train in the RVC-WebUI GUI, "
            "drop model.pth + added.index into voices/<name>/rvc/, "
            "then write meta.json by hand. Or use --method knn-vc for "
            "zero-shot cloning."
        )
