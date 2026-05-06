from __future__ import annotations

import time
from contextlib import ExitStack
from typing import Iterator

import numpy as np

from ._log import get_logger
from .body import BodyRenderer
from .capture import AudioCapture, RealTimeAudioStream, WebcamCapture
from .config import PipelineConfig
from .health import preflight
from .hud import HUDState, draw_hud
from .identity import FaceSwapper
from .output import VirtualCameraOutput
from .pose import PoseTracker
from .speech import Wav2LipCorrector
from .stabilize import TemporalStabilizer
from .voice import StreamingVoiceChanger

log = get_logger("rtpfb.pipeline")


class Pipeline:
    """End-to-end orchestrator.

    Video stages (toggled via PipelineConfig):
        capture -> pose -> identity-swap -> [lipsync] -> [body] -> stabilize -> camera-out

    Audio stages (when enable_voice or enable_lipsync):
        mic -> [voice-changer] -> [virtual-mic-out]
                              \\-> lipsync queue (drives Wav2Lip)
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        run_preflight: bool = True,
        show_hud: bool = True,
    ):
        self.config = config
        self._stop = False
        self._hud = HUDState() if show_hud else None

        if run_preflight:
            report = preflight(config)
            report.raise_if_failed()

        self.capture = WebcamCapture(
            index=config.camera_index,
            width=config.width,
            height=config.height,
            fps=config.fps,
        )

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
                output_device=config.virtual_mic_device,
            )
        elif config.enable_lipsync:
            self.audio = AudioCapture(
                samplerate=config.audio_samplerate,
                channels=config.audio_channels,
                blocksize=config.audio_blocksize,
            )
        else:
            self.audio = None

        self.pose = PoseTracker() if config.enable_pose else None
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
        for _ in self.iter_frames():
            pass

    def iter_frames(self) -> Iterator[np.ndarray]:
        log.info("pipeline starting")
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
