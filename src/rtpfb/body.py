from __future__ import annotations

from typing import Optional

import numpy as np


class BodyRenderer:
    """Phase 4: pose-conditioned full-body re-render.

    Currently a passthrough. The two realistic options are:

      - Pose-conditioned diffusion (AnimateAnyone / MagicAnimate / Champ).
        Even distilled, real-time on a single GPU is borderline.
      - 3D Gaussian Splatting avatar trained per-identity.
        Real-time once trained, but training is offline + identity-specific.

    Keep this disabled (``enable_body=False``) until the realtime story is solid.
    """

    def __init__(self, target_image_path: Optional[str] = None):
        self.target_image_path = target_image_path

    def render(self, frame_rgb: np.ndarray, pose) -> np.ndarray:
        return frame_rgb
