from __future__ import annotations

import queue
from typing import Optional

import numpy as np


class WebcamCapture:
    """OpenCV webcam capture as a context manager. Yields RGB frames."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None

    def __enter__(self) -> "WebcamCapture":
        import cv2

        self._cv2 = cv2
        cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.index}")
        self._cap = cap
        return self

    def read(self) -> Optional[np.ndarray]:
        ok, frame_bgr = self._cap.read()
        if not ok:
            return None
        return self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)

    def __exit__(self, *exc) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class AudioCapture:
    """Background mic capture into a thread-safe queue."""

    def __init__(self, samplerate: int = 16000, channels: int = 1, blocksize: int = 1024):
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None

    def __enter__(self) -> "AudioCapture":
        import sounddevice as sd

        def _callback(indata, frames, time_info, status):
            self._queue.put(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=self.blocksize,
            callback=_callback,
        )
        self._stream.start()
        return self

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
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
