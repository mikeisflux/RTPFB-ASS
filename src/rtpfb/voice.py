from __future__ import annotations

import json
import queue
import socket
import struct
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .config import MODELS_DIR, THIRD_PARTY_DIR


# Target end-to-end latency: 115ms.
# Streaming chunk granularity is the dominant tunable — small chunks reduce
# latency but raise overhead; 256 samples @ 16kHz = 16ms per chunk.
DEFAULT_CHUNK_SAMPLES = 256
DEFAULT_SAMPLE_RATE = 16000
# Crossfade between consecutive output chunks to hide model boundary clicks.
DEFAULT_CROSSFADE_SAMPLES = 64


class StreamingVoiceChanger:
    """Real-time voice-to-voice conversion.

    Two backends, picked by the ``backend`` arg:

      * ``"rvc"`` — in-process RVC inference. Loads the RVC generator + a
        f0 estimator (RMVPE) + HuBERT content encoder. Lowest latency once
        the models are warm; needs the weights laid out under
        ``models/voice/<voice_name>/``.

      * ``"w-okada"`` — remote backend. Talks to a running
        ``third_party/voice-changer`` server (default ``ws://localhost:18888``).
        Lets you reuse w-okada's already-tuned realtime stack today while the
        in-process path is being built.

    The streaming contract is a single method: ``process(chunk) -> chunk``.
    Drop-in between mic capture and Wav2Lip / virtual-mic output. Internal
    overlap-add hides chunk boundaries.
    """

    def __init__(
        self,
        voice_model: str,
        backend: str = "rvc",
        device: str = "cuda",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_samples: int = DEFAULT_CHUNK_SAMPLES,
        crossfade_samples: int = DEFAULT_CROSSFADE_SAMPLES,
        ws_url: str = "ws://localhost:18888",
        pitch_shift_semitones: float = 0.0,
    ):
        self.voice_model = voice_model
        self.backend = backend
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.crossfade_samples = crossfade_samples
        self.pitch_shift_semitones = pitch_shift_semitones
        self.ws_url = ws_url

        self._tail = np.zeros(crossfade_samples, dtype=np.float32)

        if backend == "rvc":
            self._init_rvc()
        elif backend == "w-okada":
            self._init_okada()
        else:
            raise ValueError(f"Unknown voice-changer backend: {backend!r}")

    def _init_rvc(self) -> None:
        """In-process RVC inference path.

        Pieces (each lazy-loaded so the import doesn't pay for what's unused):

          * ContentVec / HuBERT encoder — extracts speaker-independent content
            embeddings every 16ms-ish frame.
          * RMVPE f0 estimator — pitch contour for the chunk.
          * RVC generator (synth_t) — produces target-voice audio from
            (content, f0, target speaker embedding).

        We path-import from ``third_party/RVC-WebUI`` rather than vendoring
        because the model registry + checkpoint loaders move around between
        RVC versions.
        """
        rvc_root = THIRD_PARTY_DIR / "RVC-WebUI"
        if not rvc_root.exists():
            raise FileNotFoundError(
                f"{rvc_root} missing — run scripts/install_third_party.sh"
            )
        # Lazy: actual model load happens on first ``process`` call so the
        # constructor stays cheap and the model loads on the audio thread.
        self._rvc_root = rvc_root
        self._rvc = None  # populated by _ensure_rvc_loaded

    def _ensure_rvc_loaded(self) -> None:
        if self._rvc is not None:
            return
        import sys

        if str(self._rvc_root) not in sys.path:
            sys.path.insert(0, str(self._rvc_root))

        # NOTE: RVC-WebUI's public API has churned. The pieces we need at
        # inference time are roughly:
        #   from infer.modules.vc.modules import VC
        #   from configs.config import Config
        # Wire the exact symbols once we lock to a specific RVC release tag.
        raise NotImplementedError(
            "In-process RVC backend not wired yet. Use backend='w-okada' "
            "against a running third_party/voice-changer server, or pin RVC "
            "and finish the loader."
        )

    def _init_okada(self) -> None:
        """Connect to a running w-okada/voice-changer instance.

        Run the server first:

            cd third_party/voice-changer/server
            python MMVCServerSIO.py -p 18888 --https false

        Load the target voice in its UI (or via its API) and copy the slot
        index into PipelineConfig.voice_slot.
        """
        try:
            import websocket  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "websocket-client is required for the w-okada backend. "
                "Install with: pip install websocket-client"
            ) from e
        self._websocket = websocket
        self._ws = websocket.create_connection(self.ws_url, timeout=5.0)

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Convert a chunk of source-voice audio to target-voice audio.

        ``audio_chunk`` is mono float32 at ``self.sample_rate``. The returned
        chunk is the same length and dtype.
        """
        if audio_chunk.size == 0:
            return audio_chunk

        if self.backend == "w-okada":
            converted = self._process_okada(audio_chunk)
        else:
            self._ensure_rvc_loaded()
            converted = self._process_rvc(audio_chunk)

        return self._crossfade(converted)

    def _process_okada(self, chunk: np.ndarray) -> np.ndarray:
        # w-okada's protocol expects int16 PCM frames over WebSocket.
        pcm = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        self._ws.send_binary(pcm)
        reply = self._ws.recv()
        if isinstance(reply, str):
            reply = reply.encode()
        out_int16 = np.frombuffer(reply, dtype=np.int16)
        return (out_int16.astype(np.float32) / 32767.0)

    def _process_rvc(self, chunk: np.ndarray) -> np.ndarray:  # noqa: ARG002
        raise NotImplementedError("RVC in-process backend not wired yet")

    def _crossfade(self, chunk: np.ndarray) -> np.ndarray:
        """Linear crossfade with the previous tail to suppress chunk-edge clicks."""
        if chunk.size <= self.crossfade_samples:
            self._tail = chunk[-self.crossfade_samples:]
            return chunk
        head = chunk[: self.crossfade_samples]
        ramp = np.linspace(0.0, 1.0, self.crossfade_samples, dtype=np.float32)
        head = head * ramp + self._tail * (1.0 - ramp)
        out = np.concatenate([head, chunk[self.crossfade_samples:]])
        self._tail = chunk[-self.crossfade_samples:].copy()
        return out

    def close(self) -> None:
        if self.backend == "w-okada" and getattr(self, "_ws", None) is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
