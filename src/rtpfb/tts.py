from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .config import MODELS_DIR, THIRD_PARTY_DIR


class OpenVoiceTTS:
    """Text-to-speech via OpenVoice (path-imported from third_party/OpenVoice).

    OpenVoice gives us voice-cloning TTS: we synthesize from a base TTS model,
    then convert the tone to match a reference voice clip.

    Run ``scripts/install_third_party.sh`` to clone OpenVoice, and place its
    pretrained checkpoints under ``models/openvoice/``.
    """

    def __init__(
        self,
        reference_voice_path: str,
        language: str = "English",
        speed: float = 1.0,
        device: str = "cuda",
    ):
        ov_path = THIRD_PARTY_DIR / "OpenVoice"
        if not ov_path.exists():
            raise FileNotFoundError(
                f"{ov_path} missing — run scripts/install_third_party.sh"
            )
        if str(ov_path) not in sys.path:
            sys.path.insert(0, str(ov_path))

        from openvoice.api import BaseSpeakerTTS, ToneColorConverter
        from openvoice import se_extractor

        ckpt_root = MODELS_DIR / "openvoice"
        base_ckpt = ckpt_root / "base_speakers" / "EN"
        converter_ckpt = ckpt_root / "converter"

        self._tts = BaseSpeakerTTS(str(base_ckpt / "config.json"), device=device)
        self._tts.load_ckpt(str(base_ckpt / "checkpoint.pth"))

        self._converter = ToneColorConverter(str(converter_ckpt / "config.json"), device=device)
        self._converter.load_ckpt(str(converter_ckpt / "checkpoint.pth"))

        self._source_se = self._tts.hps.data.spk2id["default"]  # placeholder; OpenVoice exposes this differently per version
        self._target_se, _ = se_extractor.get_se(
            reference_voice_path,
            self._converter,
            target_dir=str(MODELS_DIR / "openvoice" / "processed"),
            vad=True,
        )
        self.language = language
        self.speed = speed
        self._procs_dir = MODELS_DIR / "openvoice" / "tmp"
        self._procs_dir.mkdir(parents=True, exist_ok=True)

    def synth(self, text: str) -> np.ndarray:
        """Synthesize ``text`` and return a mono float32 waveform at 16 kHz."""
        import soundfile as sf

        tmp_path = self._procs_dir / "tmp.wav"
        out_path = self._procs_dir / "out.wav"

        self._tts.tts(text, str(tmp_path), speaker="default", language=self.language, speed=self.speed)
        self._converter.convert(
            audio_src_path=str(tmp_path),
            src_se=self._source_se,
            tgt_se=self._target_se,
            output_path=str(out_path),
            message="@rtpfb",
        )
        wav, sr = sf.read(str(out_path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
        return wav


class TTSAudioSource:
    """Drop-in replacement for ``AudioCapture`` driven by an ``OpenVoiceTTS``.

    Producer thread runs synthesis; consumer reads chunks at the same cadence
    as the mic-driven path so the rest of the pipeline doesn't care which
    audio source is wired up.
    """

    def __init__(self, tts: OpenVoiceTTS, samplerate: int = 16000, chunk_size: int = 1024):
        self.tts = tts
        self.samplerate = samplerate
        self.channels = 1
        self.chunk_size = chunk_size
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._closed = threading.Event()

    def __enter__(self) -> "TTSAudioSource":
        return self

    def queue_text(self, text: str) -> None:
        wav = self.tts.synth(text)
        for start in range(0, len(wav), self.chunk_size):
            self._queue.put(wav[start : start + self.chunk_size].reshape(-1, 1))

    def read_chunks(self, max_chunks: int = 16) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for _ in range(max_chunks):
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return np.zeros((0, self.channels), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def __exit__(self, *exc) -> None:
        self._closed.set()
