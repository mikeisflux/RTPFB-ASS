from __future__ import annotations

import time
from contextlib import ExitStack
from typing import Iterator

import numpy as np

from ._log import get_logger
from .capture import AudioCapture, RealTimeAudioStream, WebcamCapture
from .config import PipelineConfig
from .errors import ConfigurationError
from .health import preflight
from .hud import HUDState, draw_hud
from .pose import PoseTracker
from .voice import StreamingVoiceChanger

log = get_logger("rtpfb.pipeline")


class Pipeline:
    """End-to-end orchestrator.

    Two modes:

      mode="mocap"
          Mediapipe Holistic → VMC over UDP → external 3D renderer
          (Unreal + MetaHuman + EVMC4U, Unity VMC4U, VSeeFace…). The
          renderer handles avatar mesh + physics + final video output;
          we only ship pose data and converted voice audio.

      mode="faceswap"
          Legacy 2D path: capture → pose → InsightFace face swap →
          [Wav2Lip] → stabilize → virtual camera. No body, no physics.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        run_preflight: bool = True,
        show_hud: bool = True,
    ):
        if config.mode not in {"mocap", "faceswap"}:
            raise ConfigurationError(f"Unknown mode {config.mode!r}; expected 'mocap' or 'faceswap'.")

        self.config = config
        self._stop = False
        self._hud = HUDState() if show_hud and config.mode == "faceswap" else None

        if run_preflight:
            preflight(config).raise_if_failed()

        # Visual stages are only set up if some visual feature is requested.
        # Voice-only configs skip camera, pose, and mode-specific init entirely.
        needs_visual = (
            config.enable_pose
            or config.mode == "faceswap"
            or config.enable_lipsync
            or config.output_virtual_camera
            or config.enable_body
        )

        if needs_visual:
            self.capture = WebcamCapture(
                index=config.camera_index,
                width=config.width,
                height=config.height,
                fps=config.fps,
            )
            self.pose = PoseTracker() if config.enable_pose else None
            if config.mode == "mocap":
                self._init_mocap_stages(config)
            else:
                self._init_faceswap_stages(config)
        else:
            self.capture = None
            self.pose = None
            self._init_no_visual_stages()

        # Audio + voice are mode-agnostic.
        self.voice = (
            StreamingVoiceChanger(
                voice_model=config.voice_model,
                backend=config.voice_backend,
                sample_rate=config.audio_samplerate,
                chunk_samples=config.audio_blocksize,
                crossfade_samples=config.voice_crossfade_samples,
                ws_url=config.voice_ws_url,
                pitch_shift_semitones=config.voice_pitch_shift,
            )
            if config.enable_voice
            else None
        )

        if config.enable_voice or config.output_virtual_mic:
            self.audio = RealTimeAudioStream(
                voice_changer=self.voice,
                samplerate=config.audio_samplerate,
                channels=config.audio_channels,
                blocksize=config.audio_blocksize,
                input_device=config.mic_device,
                output_device=config.virtual_mic_device,
            )
        elif config.enable_lipsync:
            self.audio = AudioCapture(
                samplerate=config.audio_samplerate,
                channels=config.audio_channels,
                blocksize=config.audio_blocksize,
                device=config.mic_device,
            )
        else:
            self.audio = None

    def _init_mocap_stages(self, config: PipelineConfig) -> None:
        from .config import MODELS_DIR
        from .vmc import VMCSender

        self.vmc = VMCSender(host=config.vmc_host, port=config.vmc_port)
        self.face_blendshapes = None
        if config.vmc_face_blendshapes:
            try:
                from .pose import FaceBlendshapeExtractor

                self.face_blendshapes = FaceBlendshapeExtractor(
                    model_path=MODELS_DIR / config.face_landmarker_model
                )
            except Exception as exc:  # noqa: BLE001 — fall back to synthesised blendshapes
                log.warning(
                    "Face Landmarker v2 unavailable (%s); falling back to "
                    "synthesised jawOpen/eyeBlink only.", exc
                )
        self.identity = None
        self.lipsync = None
        self.body = None
        self.stabilizer = None
        self.output = None  # The 3D renderer owns video output.

    def _init_no_visual_stages(self) -> None:
        """Voice/audio-only configs: no camera, no pose, no mode-specific
        stages. All visual attributes are None so the pipeline's visual
        helpers are inert."""
        self.vmc = None
        self.face_blendshapes = None
        self.identity = None
        self.lipsync = None
        self.body = None
        self.stabilizer = None
        self.output = None

    def _init_faceswap_stages(self, config: PipelineConfig) -> None:
        from .body import BodyRenderer
        from .identity import FaceSwapper
        from .output import VirtualCameraOutput
        from .speech import Wav2LipCorrector
        from .stabilize import TemporalStabilizer

        self.vmc = None
        self.identity = FaceSwapper(config.target_face_path, config.swap_model)
        self.lipsync = Wav2LipCorrector() if config.enable_lipsync else None
        self.body = BodyRenderer(config.target_face_path) if config.enable_body else None
        self.stabilizer = (
            TemporalStabilizer(blend=config.stabilize_blend) if config.enable_stabilize else None
        )
        self.output = (
            VirtualCameraOutput(config.width, config.height, config.fps)
            if config.output_virtual_camera
            else None
        )

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        if self.capture is None:
            self._run_voice_only()
            return
        for _ in self.iter_frames():
            pass

    def _run_voice_only(self) -> None:
        """Audio-only run loop: enter the audio context and idle until
        ``request_stop()`` is called. The audio thread (sounddevice
        callback) does all the work."""
        log.info("pipeline starting (voice-only, no video)")
        try:
            with ExitStack() as stack:
                if self.audio is not None:
                    stack.enter_context(self.audio)
                log.info("audio streaming - press Ctrl+C to stop")
                while not self._stop:
                    time.sleep(0.1)
        finally:
            if self.voice is not None:
                self.voice.close()
            log.info("pipeline shut down")

    def iter_frames(self) -> Iterator[np.ndarray]:
        log.info("pipeline starting (mode=%s)", self.config.mode)
        try:
            with ExitStack() as stack:
                cap = stack.enter_context(self.capture)
                audio = stack.enter_context(self.audio) if self.audio is not None else None
                cam_out = stack.enter_context(self.output) if self.output is not None else None

                t0 = time.time()
                n = 0
                while not self._stop:
                    frame = cap.read()
                    if frame is None:
                        log.warning("capture returned None, ending stream")
                        break

                    pose_data = self.pose.process(frame) if self.pose is not None else None

                    if self.config.mode == "mocap":
                        frame = self._step_mocap(frame, pose_data)
                    else:
                        frame = self._step_faceswap(frame, pose_data, audio)

                    if cam_out is not None:
                        cam_out.send(frame)

                    yield frame

                    n += 1
                    if n % 60 == 0:
                        dt = time.time() - t0
                        log.info("%d frames, %.1f fps", n, n / dt)
        finally:
            if self.voice is not None:
                self.voice.close()
            log.info("pipeline shut down")

    def _step_mocap(self, frame: np.ndarray, pose_data) -> np.ndarray:
        if self.vmc is not None and pose_data is not None:
            blendshapes = None
            if self.config.vmc_face_blendshapes:
                if self.face_blendshapes is not None:
                    try:
                        blendshapes = self.face_blendshapes.extract(frame)
                    except Exception as exc:  # noqa: BLE001
                        log.debug("Face Landmarker extract failed: %s", exc)
                        blendshapes = None
                if not blendshapes:
                    from .vmc import blendshapes_from_face_landmarks

                    blendshapes = blendshapes_from_face_landmarks(pose_data.face_landmarks)
            self.vmc.send(pose_data, blendshapes=blendshapes)
        # In mocap mode the local frame is just used for HUD / preview.
        return frame

    def _step_faceswap(self, frame: np.ndarray, pose_data, audio) -> np.ndarray:
        frame = self.identity.swap(frame, pose=pose_data)

        if self.lipsync is not None and audio is not None:
            frame = self.lipsync(frame, audio.read_chunks(), pose=pose_data)

        if self.body is not None and pose_data is not None:
            frame = self.body.render(frame, pose_data)

        if self.stabilizer is not None:
            frame = self.stabilizer.smooth(frame)

        if self._hud is not None:
            self._hud.tick()
            self._hud.voice_state = (
                f"{self.voice.backend}:{self.voice.voice_model or '-'}"
                if self.voice is not None
                else "off"
            )
            frame = draw_hud(frame, self._hud)

        return frame
