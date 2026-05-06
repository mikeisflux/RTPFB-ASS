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
| OpenVoice TTS (text-driven mode) | `src/rtpfb/tts.py` | Skeleton — wraps `third_party/OpenVoice` for voice-cloning text-to-speech. `TTSAudioSource` is a drop-in for `AudioCapture` so you can drive the avatar from typed text. Needs OpenVoice checkpoints under `models/openvoice/` and a CLI/UX wire-up. |

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
