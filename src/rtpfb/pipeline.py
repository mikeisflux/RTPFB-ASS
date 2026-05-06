from __future__ import annotations

import time
from contextlib import ExitStack
from typing import Iterator

import numpy as np

from .body import BodyRenderer
from .capture import AudioCapture, WebcamCapture
from .config import PipelineConfig
from .identity import FaceSwapper
from .output import VirtualCameraOutput
from .pose import PoseTracker
from .speech import Wav2LipCorrector
from .stabilize import TemporalStabilizer


class Pipeline:
    """End-to-end orchestrator.

    Stages (toggled via PipelineConfig):
        capture -> pose -> identity-swap -> [lipsync] -> [body] -> stabilize -> output
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        self.capture = WebcamCapture(
            index=config.camera_index,
            width=config.width,
            height=config.height,
            fps=config.fps,
        )
        self.audio = (
            AudioCapture(
                samplerate=config.audio_samplerate,
                channels=config.audio_channels,
                blocksize=config.audio_blocksize,
            )
            if config.enable_lipsync
            else None
        )
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

    def run(self) -> None:
        for _ in self.iter_frames():
            pass

    def iter_frames(self) -> Iterator[np.ndarray]:
        with ExitStack() as stack:
            cap = stack.enter_context(self.capture)
            audio = stack.enter_context(self.audio) if self.audio is not None else None
            out = stack.enter_context(self.output) if self.output is not None else None

            t0 = time.time()
            n = 0
            while True:
                frame = cap.read()
                if frame is None:
                    break

                pose_data = self.pose.process(frame) if self.pose is not None else None
                frame = self.identity.swap(frame, pose=pose_data)

                if self.lipsync is not None and audio is not None:
                    frame = self.lipsync(frame, audio.read_chunks(), pose=pose_data)

                if self.body is not None and pose_data is not None:
                    frame = self.body.render(frame, pose_data)

                if self.stabilizer is not None:
                    frame = self.stabilizer.smooth(frame)

                if out is not None:
                    out.send(frame)

                yield frame

                n += 1
                if n % 60 == 0:
                    dt = time.time() - t0
                    print(f"[rtpfb] {n} frames, {n / dt:.1f} fps", flush=True)
