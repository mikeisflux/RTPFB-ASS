from __future__ import annotations

import numpy as np


class TemporalStabilizer:
    """Frame-to-frame blending using dense Farneback optical flow.

    Kills high-frequency flicker from the per-frame face-swap inference by
    warping the previous output into the current motion field and
    alpha-blending it with the fresh frame.

    Future work: replace the full-frame warp with a face-region warp +
    "sink-frame" reference re-anchoring (the trick from the LiveAvatar paper).
    """

    def __init__(self, blend: float = 0.35):
        import cv2

        self._cv2 = cv2
        self.blend = blend
        self._prev_gray: np.ndarray | None = None
        self._prev_out: np.ndarray | None = None

    def smooth(self, frame_rgb: np.ndarray) -> np.ndarray:
        cv2 = self._cv2
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        if self._prev_gray is None or self._prev_out is None:
            self._prev_gray = gray
            self._prev_out = frame_rgb
            return frame_rgb

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        h, w = gray.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped_prev = cv2.remap(self._prev_out, map_x, map_y, cv2.INTER_LINEAR)

        out = cv2.addWeighted(frame_rgb, 1.0 - self.blend, warped_prev, self.blend, 0.0)
        self._prev_gray = gray
        self._prev_out = out
        return out

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_out = None
