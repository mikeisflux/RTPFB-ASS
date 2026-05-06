from __future__ import annotations

import sys
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

    Backends (set via ``backend`` or auto-detected from the voice library):

      * ``"knn-vc"`` — in-process KNN-VC. Loads cached WavLM features for the
        target voice (created with ``rtpfb voice add --method knn-vc``).
        Zero-shot: no training run, ~1–5 min of reference audio is enough.

      * ``"rvc"`` — in-process RVC inference. Highest quality. Loader is a
        stub until we pin an RVC release tag.

      * ``"w-okada"`` — talks to a running ``third_party/voice-changer``
        server over WebSocket. Lets you reuse w-okada's tuned realtime stack
        today; voice slot IDs map to ``--voice-model``.

    Streaming contract is one method: ``process(chunk) -> chunk``. Drop in
    between mic capture and the virtual mic / Wav2Lip lipsync feed. An
    internal crossfade hides chunk boundaries.
    """

    def __init__(
        self,
        voice_model: str = "",
        backend: str = "auto",
        device: str = "cuda",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_samples: int = DEFAULT_CHUNK_SAMPLES,
        crossfade_samples: int = DEFAULT_CROSSFADE_SAMPLES,
        ws_url: str = "ws://localhost:18888",
        pitch_shift_semitones: float = 0.0,
    ):
        self.voice_model = voice_model
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.crossfade_samples = crossfade_samples
        self.pitch_shift_semitones = pitch_shift_semitones
        self.ws_url = ws_url

        if backend == "auto":
            backend = self._auto_detect_backend(voice_model)
        self.backend = backend

        self._tail = np.zeros(crossfade_samples, dtype=np.float32)
        self._sio = None
        self._response_queue = None

        if backend == "knn-vc":
            self._init_knnvc()
        elif backend == "rvc":
            self._init_rvc()
        elif backend == "w-okada":
            self._init_okada()
        else:
            raise ValueError(f"Unknown voice-changer backend: {backend!r}")

    @staticmethod
    def _auto_detect_backend(voice_model: str) -> str:
        if not voice_model:
            return "w-okada"
        try:
            from .voice_library import VoiceLibrary

            meta = VoiceLibrary().get(voice_model)
            return meta.method
        except KeyError:
            return "w-okada"

    # --------------------------- KNN-VC backend ---------------------------

    def _init_knnvc(self) -> None:
        from .voice_library import VoiceLibrary

        lib = VoiceLibrary()
        artifact = lib.artifact_path(self.voice_model)
        if not artifact.exists():
            raise FileNotFoundError(
                f"KNN-VC features not found: {artifact}. "
                f"Run `rtpfb voice add {self.voice_model} --samples DIR --method knn-vc` first."
            )

        knn_root = THIRD_PARTY_DIR / "knn-vc"
        if not knn_root.exists():
            raise FileNotFoundError(
                f"{knn_root} missing — run scripts/install_third_party.sh"
            )
        if str(knn_root) not in sys.path:
            sys.path.insert(0, str(knn_root))

        import torch

        self._torch = torch
        self._knn = torch.hub.load("bshall/knn-vc", "knn_vc", trust_repo=True, prematched=True)
        self._target_feats = torch.load(artifact, map_location=self.device)

        # Streaming buffer. KNN-VC needs ~500ms+ of context for stable WavLM
        # features; we pad with zeros until full, then keep a sliding 2s window.
        self._knnvc_buffer = np.zeros(0, dtype=np.float32)
        self._knnvc_min_samples = int(self.sample_rate * 0.5)
        self._knnvc_window_samples = int(self.sample_rate * 1.0)
        self._knnvc_max_buffer = int(self.sample_rate * 2.0)

    def _process_knnvc(self, chunk: np.ndarray) -> np.ndarray:
        self._knnvc_buffer = np.concatenate([self._knnvc_buffer, chunk])
        if self._knnvc_buffer.size > self._knnvc_max_buffer:
            self._knnvc_buffer = self._knnvc_buffer[-self._knnvc_max_buffer:]

        if self._knnvc_buffer.size < self._knnvc_min_samples:
            return chunk

        torch = self._torch
        ctx = self._knnvc_buffer[-self._knnvc_window_samples:]
        with torch.no_grad():
            wav_t = torch.from_numpy(ctx).unsqueeze(0).to(self._target_feats.device)
            query_feats = self._knn.get_features(wav_t)
            out_wav = self._knn.match(query_feats, self._target_feats, topk=4)
            out_np = out_wav.detach().cpu().numpy().astype(np.float32)

        if out_np.size < chunk.size:
            pad = np.zeros(chunk.size - out_np.size, dtype=np.float32)
            return np.concatenate([pad, out_np])
        return out_np[-chunk.size:]

    # ----------------------------- RVC backend ----------------------------

    def _init_rvc(self) -> None:
        rvc_root = THIRD_PARTY_DIR / "RVC-WebUI"
        if not rvc_root.exists():
            raise FileNotFoundError(
                f"{rvc_root} missing — run scripts/install_third_party.sh"
            )
        self._rvc_root = rvc_root
        self._rvc = None

    def _process_rvc(self, chunk: np.ndarray) -> np.ndarray:  # noqa: ARG002
        raise NotImplementedError(
            "In-process RVC backend not wired yet. Use --voice-backend w-okada "
            "with a running voice-changer server, or --voice-backend knn-vc "
            "with a knn-vc voice from the library."
        )

    # --------------------------- w-okada backend --------------------------

    def _init_okada(self) -> None:
        # w-okada speaks Socket.IO (not raw WebSocket). Audio streaming uses
        # an event-based protocol: client emits `request_message` with
        # [timestamp_ms, int16_pcm_bytes], server replies asynchronously by
        # emitting `response` with [timestamp_ms, int16_pcm_bytes, perf].
        try:
            import socketio  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "python-socketio[client] is required for the w-okada backend. "
                'Install with: pip install "python-socketio[client]"'
            ) from exc

        from queue import Queue

        self._socketio = socketio
        self._sio = socketio.Client()
        self._response_queue = Queue()

        @self._sio.on("response")
        def _on_response(*args):
            # w-okada server may emit either as separate positional args or
            # as a single list - normalize to a tuple.
            self._response_queue.put(args)

        # python-socketio.Client expects an http(s) URL. Auto-translate the
        # ws:// URL the user passes (matches the historical --voice-ws flag).
        url = self.ws_url
        if url.startswith("ws://"):
            url = "http://" + url[len("ws://"):]
        elif url.startswith("wss://"):
            url = "https://" + url[len("wss://"):]
        self._sio.connect(url, transports=["websocket"], wait_timeout=10.0)

    def _process_okada(self, chunk: np.ndarray) -> np.ndarray:
        import time
        from queue import Empty

        # Drain stale responses left over from a previous timed-out request,
        # so we never return audio that doesn't belong to this chunk.
        while not self._response_queue.empty():
            try:
                self._response_queue.get_nowait()
            except Empty:
                break

        pcm = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        timestamp_ms = int(time.time() * 1000)
        self._sio.emit("request_message", [timestamp_ms, pcm])

        try:
            response = self._response_queue.get(timeout=2.0)
        except Empty:
            return chunk  # pass-through on timeout - keeps audio flowing

        # Normalize: server may have emitted either (ts, bin, perf) as 3 args
        # or [ts, bin, perf] as a single list arg.
        if len(response) == 1 and isinstance(response[0], (list, tuple)):
            response = response[0]
        if len(response) < 2:
            return chunk
        out_bytes = response[1]
        if isinstance(out_bytes, str):
            out_bytes = out_bytes.encode("latin-1")
        out_int16 = np.frombuffer(out_bytes, dtype=np.int16)
        return out_int16.astype(np.float32) / 32767.0

    # --------------------------- common surface ---------------------------

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        if audio_chunk.size == 0:
            return audio_chunk

        if self.backend == "knn-vc":
            converted = self._process_knnvc(audio_chunk)
        elif self.backend == "rvc":
            converted = self._process_rvc(audio_chunk)
        else:
            converted = self._process_okada(audio_chunk)

        return self._crossfade(converted)

    def _crossfade(self, chunk: np.ndarray) -> np.ndarray:
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
        if self._sio is not None:
            try:
                self._sio.disconnect()
            except Exception:
                pass
            self._sio = None
