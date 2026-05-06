from __future__ import annotations

import numpy as np


class Wav2LipCorrector:
    """Phase 2: post-process the swapped frame with Wav2Lip lip-sync correction.

    Currently a passthrough. A real implementation needs:

      1. ``third_party/Wav2Lip`` cloned (``scripts/install_third_party.sh``)
      2. ``models/wav2lip_gan.pth`` downloaded
      3. A short windowed audio buffer (~200ms) of mel features
      4. ROI extraction around the mouth, then merge back over the swapped face

    Real-time Wav2Lip is the tricky bit — the original code is offline-batch.
    Practical paths:

      - Run inference every N frames and warp/blend between
      - Use a distilled / streaming variant
      - Replace Wav2Lip with VideoReTalking or MuseTalk for lower latency
    """

    def __init__(self, checkpoint: str = "wav2lip_gan.pth", window_ms: int = 200):
        self.checkpoint = checkpoint
        self.window_ms = window_ms
        self._model = None

    def __call__(self, frame_rgb: np.ndarray, audio_chunk: np.ndarray) -> np.ndarray:
        return frame_rgb
