from __future__ import annotations

from typing import Optional

import numpy as np


class VirtualCameraOutput:
    """Send RGB frames to a virtual camera so OBS / Zoom / browsers can pick them up.

    Backends:
      - Linux: needs the v4l2loopback kernel module loaded.
      - Windows: needs the OBS virtual camera driver installed.
      - macOS: limited support; see pyvirtualcam docs.
    """

    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self._cam = None

    def __enter__(self) -> "VirtualCameraOutput":
        import pyvirtualcam

        self._cam = pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps)
        return self

    def send(self, frame_rgb: np.ndarray) -> None:
        if frame_rgb.shape[:2] != (self.height, self.width):
            import cv2

            frame_rgb = cv2.resize(frame_rgb, (self.width, self.height))
        self._cam.send(frame_rgb)
        self._cam.sleep_until_next_frame()

    def __exit__(self, *exc) -> None:
        if self._cam is not None:
            self._cam.close()
            self._cam = None


class VirtualMicOutput:
    """Push converted audio to a loopback / virtual audio device so OBS,
    Zoom, etc. pick it up as a microphone.

    Routing per OS:
      - Linux: load `module-null-sink` in PulseAudio / PipeWire and select the
        sink's monitor as the input source in the consumer app. The
        ``device`` arg accepts a sounddevice index or device name substring.
      - Windows: install VB-CABLE; pass ``device="CABLE Input"`` (or the index).
      - macOS: install BlackHole; pass ``device="BlackHole 2ch"``.
    """

    def __init__(
        self,
        samplerate: int = 16000,
        channels: int = 1,
        blocksize: int = 256,
        device: Optional[str | int] = None,
    ):
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.device = device
        self._stream = None

    def __enter__(self) -> "VirtualMicOutput":
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype="float32",
            device=self.device,
        )
        self._stream.start()
        return self

    def send(self, audio_chunk: np.ndarray) -> None:
        if audio_chunk.size == 0:
            return
        if audio_chunk.ndim == 1:
            audio_chunk = audio_chunk[:, np.newaxis]
        self._stream.write(np.ascontiguousarray(audio_chunk, dtype=np.float32))

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
