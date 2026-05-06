# Build Status

Tracks what's implemented vs. stubbed against the whitepaper phases in `README.md`.

## Phase 1 — Face swap MVP → virtual camera

| Component | File | Status |
|---|---|---|
| Webcam capture | `src/rtpfb/capture.py` | Done |
| Audio capture | `src/rtpfb/capture.py` | Done |
| Face swap (InsightFace inswapper) | `src/rtpfb/identity.py` | Done — needs model weights |
| Virtual camera output | `src/rtpfb/output.py` | Done — needs OBS / v4l2loopback backend |
| Pipeline orchestrator | `src/rtpfb/pipeline.py` | Done |
| CLI | `src/rtpfb/cli.py` | Done |

## Phase 2 — Speech-driven lip sync

| Component | File | Status |
|---|---|---|
| Wav2Lip inference (vendored) | `src/rtpfb/_vendor/wav2lip/` | Done — `audio.py`, `hparams.py`, `models/{conv,wav2lip}.py` lifted from Rudrabha/Wav2Lip with imports rewritten relative. |
| Streaming Wav2Lip post-processor | `src/rtpfb/speech.py` | Done — buffers audio, computes mel, crops face via MediaPipe landmarks, runs the model per frame, composites the lower-half output back. Falls back to passthrough if the checkpoint is missing or no face is found. |
## Phase 2b — Real-time voice-to-voice conversion (target ≤115ms latency)

| Component | File | Status |
|---|---|---|
| Streaming voice changer (3 backends) | `src/rtpfb/voice.py` | `knn-vc` in-process path **wired** (loads features from the voice library, streams via WavLM + HiFi-GAN); `w-okada` WebSocket path **wired** (talks to a running voice-changer server); `rvc` in-process **stub** (loader pending an RVC release pin). |
| Realtime audio bridge | `src/rtpfb/capture.py::RealTimeAudioStream` | Done — sounddevice callback runs voice conversion in the audio thread, ~16ms buffering. |
| Virtual microphone output | `src/rtpfb/output.py::VirtualMicOutput` | Done — sounddevice OutputStream into a loopback device (PulseAudio null-sink / VB-CABLE / BlackHole). |
| Latency budget | — | capture 16ms + HuBERT/WavLM 10–20ms + f0 5–10ms + generator 30–50ms + output 16ms ≈ **75–115ms** with crossfade between chunks. |

## Voice library (clone a voice from samples)

| Component | File | Status |
|---|---|---|
| File-system voice registry | `src/rtpfb/voice_library.py::VoiceLibrary` | Done — `voices/<name>/{samples,meta.json,knnvc/features.pt or rvc/}` layout. |
| KNN-VC embedder (zero-shot) | `src/rtpfb/voice_library.py::KNNVCEmbedder` | Done — caches WavLM features from training audio. ~1–5 min of clean speech is enough. |
| RVC trainer | `src/rtpfb/voice_library.py::RVCTrainer` | **Stub** — surfaces a clear error. Train via the RVC-WebUI GUI for now and drop the `.pth` + `.index` into `voices/<name>/rvc/`. |
| CLI (`rtpfb voice add/list/remove`) | `src/rtpfb/cli.py` | Done — subparser-based CLI; the streamer reads `--voice-model <name>` and auto-picks the backend from `meta.json`. |

## OBS streaming

The pipeline already pipes to OBS via `pyvirtualcam` (video) + a sounddevice
loopback device (audio). See **`STREAMING.md`** for the full setup steps —
loopback audio device per OS, OBS source wiring, A/V sync offsets, and lower-latency
NDI / direct-RTMP options for later.

## Phase 3 — Pose tracking

| Component | File | Status |
|---|---|---|
| MediaPipe Holistic wrapper | `src/rtpfb/pose.py` | Done — produces `PoseFrame` with face / body / hand landmarks |
| Pose overlay / consumer | — | Pose is captured but downstream stages don't condition on it yet |

## Phase 4 — Pose-conditioned body render

| Component | File | Status |
|---|---|---|
| AnimateAnyone / MagicAnimate hook | `src/rtpfb/body.py` | **Stub** (passthrough). Real-time on a single 5090 is borderline — distill or skip. |

## Phase 5 — Temporal stabilization

| Component | File | Status |
|---|---|---|
| Optical-flow blender | `src/rtpfb/stabilize.py` | Done — basic Farneback + alpha blend. Sink-frame reuse not yet wired. |

## Quickstart

```bash
# 1. Install Python deps
pip install -e .

# 2. Pull upstream repos (DeepFaceLive, FaceFusion, Wav2Lip, AnimateAnyone, MagicAnimate)
bash scripts/install_third_party.sh

# 3. Download model weights (manual — see notes inside)
bash scripts/download_models.sh

# 4. Run with a target face image
rtpfb --target ./assets/target.png
```

On Linux you need `v4l2loopback` for the virtual camera; on Windows the OBS virtual camera driver. See `pyvirtualcam` docs for setup.

## Known gaps

- The InsightFace `inswapper_128.onnx` model is non-commercial use only. For commercial work, swap in a pipeline you have the rights to (e.g. a model you trained, or a licensed FaceFusion build).
- The MVP face swap engine is a thin wrapper around InsightFace. Swapping in DeepFaceLive (subprocess) or FaceFusion (subprocess) is on the roadmap; the `FaceSwapper` interface is intentionally narrow so the engine is pluggable.
- No tests yet. Adding a fixture-based test for each stage with synthetic frames is the next infra task.
