# Build Status

Tracks what's implemented vs. stubbed. The project has **two operating modes**;
the recommended one is *mocap*.

## Mode: mocap (default — full-body + physics)

Our Python pipeline does mocap + voice; an external 3D renderer (Unreal 5 +
MetaHuman + EVMC4U recommended, or Unity / VSeeFace for VRM) does avatar
rendering + soft-body physics + final video output.

| Component | File | Status |
|---|---|---|
| Body / face / hand mocap | `src/rtpfb/pose.py` | Done — MediaPipe Holistic |
| Quaternion math util | `src/rtpfb/_math.py` | Done — quat_from_two_vectors / quat_from_basis / quat_mul / quat_rotate. 12 unit tests. |
| VMC protocol output (OSC/UDP) | `src/rtpfb/vmc.py` | Done — root + 22 body bones + 30 finger bones, all with computed rotations. ARKit blendshapes (52 from Face Landmarker v2 if available, 3 synthesised fallback). |
| Body bone rotations | `src/rtpfb/vmc.py::_compute_body_transforms` | Done — world-space rotations from rest direction → live direction per bone. Spine derived from hip↔shoulder axis; head from ear-line + nose; arms / legs per segment. |
| Hand finger tracking | `src/rtpfb/vmc.py::_compute_finger_transforms` | Done — 5 fingers × 3 phalanges × 2 hands = 30 bones, each with computed rotation. |
| Face Landmarker v2 (52 ARKit blendshapes) | `src/rtpfb/pose.py::FaceBlendshapeExtractor` | Done — wraps MediaPipe `tasks.vision.FaceLandmarker`. Pipeline auto-detects, falls back to synthesised 3-blendshape set if the `.task` file isn't present. `scripts/download_models.sh` auto-fetches it (Apache-2.0). |
| Mocap-mode pipeline orchestration | `src/rtpfb/pipeline.py` | Done — `mode="mocap"` skips face-swap/Wav2Lip and ships pose data. |
| Unreal + MetaHuman setup | `UNREAL.md` | Done — full setup walkthrough. EVMC4U plugin handles VMC → MetaHuman skeleton mapping. |
| Live Face Landmarker as a separate frame | — | Future — currently the v2 task runs synchronously per frame on the same RGB buffer. If FPS suffers, move to async with the LIVE_STREAM running mode. |

## Mode: faceswap (legacy 2D path)

Kept as a fallback for users who can't / won't set up Unreal. No body, no
physics — just a 2D face swap to a virtual camera.

| Component | File | Status |
|---|---|---|
| Webcam capture | `src/rtpfb/capture.py` | Done |
| Audio capture | `src/rtpfb/capture.py` | Done |
| Face swap (InsightFace inswapper) | `src/rtpfb/identity.py` | Done — needs `inswapper_128.onnx` |
| Wav2Lip lip sync | `src/rtpfb/speech.py` | Done — needs `wav2lip_gan.pth` |
| Optical-flow stabilizer | `src/rtpfb/stabilize.py` | Done |
| Virtual camera output | `src/rtpfb/output.py::VirtualCameraOutput` | Done — pyvirtualcam |
| Body re-render (AnimateAnyone-style) | `src/rtpfb/body.py` | Stub — will not be finished; this lives in mocap mode now. |

## Voice (mode-agnostic, target ≤115ms latency)

| Component | File | Status |
|---|---|---|
| Streaming voice changer (3 backends) | `src/rtpfb/voice.py` | knn-vc and w-okada wired; rvc loader is a stub pending RVC release pin. |
| Voice library (clone from samples) | `src/rtpfb/voice_library.py` | Done — knn-vc embedder zero-shot; rvc training delegates to RVC-WebUI GUI. |
| Realtime audio bridge | `src/rtpfb/capture.py::RealTimeAudioStream` | Done — sounddevice callback runs voice conversion in audio thread. |
| Virtual mic output | `src/rtpfb/output.py::VirtualMicOutput` | Done — loopback device (PulseAudio null-sink / VB-CABLE / BlackHole). |
| CLI | `src/rtpfb/cli.py` | Done — `rtpfb voice add/list/remove`, `--voice-model`. |

## Production foundations

| | Status |
|---|---|
| Structured logging (`_log.py`) | Done — JSON or human, RTPFB_LOG_LEVEL env |
| Typed exception hierarchy (`errors.py`) | Done |
| Preflight health checks (`health.py`) | Done — `rtpfb health` subcommand |
| YAML preset loader (`preset.py`) | Done — `--preset path.yaml` |
| HUD overlay (`hud.py`) | Done — fps + voice state on faceswap output |
| Signal handling (graceful shutdown) | Done — SIGINT/SIGTERM trip ExitStack cleanup |
| Tests | 38+ unit tests, no GPU/audio/camera required |
| ruff + mypy + pytest configs | Done — all in `pyproject.toml` |
| GitHub Actions CI | Done — lint + matrix test on Python 3.10/3.11/3.12 |
| Pre-commit hooks | Done |
| Dockerfile (CUDA 12 runtime) | Done |
| Makefile | Done |
| LICENSE (MIT) | Done |
| Presets | `streaming-knnvc.yaml`, `voice-only.yaml`, `mocap-unreal.yaml` |

## Quickstart (mocap mode — recommended)

```bash
# 1. Install
pip install -e ".[voice,mocap,preset]"
bash scripts/install_third_party.sh   # voice changer dependencies

# 2. Clone a voice
rtpfb voice add me --samples ~/voice_clips/

# 3. Open Unreal (see UNREAL.md), hit Play in your MetaHuman level so
#    EVMC4U starts listening on :39539

# 4. Run the mocap + voice service
rtpfb run --preset presets/mocap-unreal.yaml --voice-model me

# 5. Open OBS, add Unreal's NDI feed as video, loopback device as audio,
#    stream to wherever.
```

## Quickstart (faceswap fallback)

```bash
rtpfb run --mode faceswap --target ./assets/face.png --voice --voice-model me
```
