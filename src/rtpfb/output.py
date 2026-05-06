from __future__ import annotations

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
