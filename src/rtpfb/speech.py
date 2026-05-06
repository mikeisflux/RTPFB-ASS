from __future__ import annotations

from typing import Optional

import numpy as np

from .config import MODELS_DIR


W2L_IMG_SIZE = 96
W2L_MEL_STEP = 16


class Wav2LipCorrector:
    """Streaming Wav2Lip lip-sync post-processor.

    Per call:
      1. Append audio chunk to the internal ring buffer.
      2. Compute a mel spectrogram for the latest ~200ms window.
      3. Crop the face region from the frame using MediaPipe landmarks.
      4. Resize to 96x96, mask lower half, concat with the reference half
         to build the 6-channel face input Wav2Lip expects.
      5. Run Wav2Lip's generator -> 96x96 face with re-synced lips.
      6. Resize back and composite the lower half over the original frame.

    Falls back to passthrough if the checkpoint is missing, the audio buffer
    is too small, or no face landmarks were provided.
    """

    def __init__(self, checkpoint: str = "wav2lip_gan.pth", device: str = "cuda"):
        ckpt_path = MODELS_DIR / checkpoint

        from ._vendor.wav2lip import audio as w2l_audio
        from ._vendor.wav2lip.hparams import hparams

        self._w2l_audio = w2l_audio
        self._hparams = hparams
        self._sr = hparams.sample_rate
        self._audio_buf = np.zeros((0,), dtype=np.float32)
        self._max_buf_samples = int(self._sr * 1.0)
        self.model = None
        self.device = device

        if not ckpt_path.exists():
            print(f"[wav2lip] checkpoint {ckpt_path} not found — running as passthrough")
            return

        import torch
        from ._vendor.wav2lip.models import Wav2Lip

        self._torch = torch
        model = Wav2Lip()
        state = torch.load(str(ckpt_path), map_location=device)
        sd = state.get("state_dict", state) if isinstance(state, dict) else state
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        self.model = model.to(device).eval()

    def __call__(self, frame_rgb: np.ndarray, audio_chunk: np.ndarray, pose=None) -> np.ndarray:
        if self.model is None:
            return frame_rgb

        if audio_chunk.size > 0:
            mono = audio_chunk[:, 0] if audio_chunk.ndim == 2 else audio_chunk
            self._audio_buf = np.concatenate([self._audio_buf, mono.astype(np.float32)])
            if self._audio_buf.size > self._max_buf_samples:
                self._audio_buf = self._audio_buf[-self._max_buf_samples:]

        min_samples = int(self._sr * 0.2)
        if self._audio_buf.size < min_samples:
            return frame_rgb

        if pose is None or getattr(pose, "face_landmarks", None) is None:
            return frame_rgb

        import cv2

        h, w = frame_rgb.shape[:2]
        lm = pose.face_landmarks
        xs = lm[:, 0] * w
        ys = lm[:, 1] * h
        x0, x1 = max(0, int(xs.min())), min(w, int(xs.max()))
        y0, y1 = max(0, int(ys.min())), min(h, int(ys.max()))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return frame_rgb

        face_crop = frame_rgb[y0:y1, x0:x1]
        face96 = cv2.resize(face_crop, (W2L_IMG_SIZE, W2L_IMG_SIZE))
        face_masked = face96.copy()
        face_masked[W2L_IMG_SIZE // 2:] = 0

        face_in = np.concatenate([face_masked, face96], axis=2)
        face_in = face_in.transpose(2, 0, 1).astype(np.float32) / 255.0
        face_in = face_in[np.newaxis]

        mel = self._w2l_audio.melspectrogram(self._audio_buf[-min_samples:])
        if mel.shape[1] < W2L_MEL_STEP:
            mel = np.pad(mel, ((0, 0), (0, W2L_MEL_STEP - mel.shape[1])), mode="edge")
        mel = mel[:, -W2L_MEL_STEP:]
        mel_in = mel[np.newaxis, np.newaxis].astype(np.float32)

        torch = self._torch
        with torch.no_grad():
            face_t = torch.from_numpy(face_in).to(self.device)
            mel_t = torch.from_numpy(mel_in).to(self.device)
            pred = self.model(mel_t, face_t)
            pred = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        pred = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
        pred_resized = cv2.resize(pred, (x1 - x0, y1 - y0))

        out = frame_rgb.copy()
        mid = (y1 - y0) // 2
        out[y0 + mid:y1, x0:x1] = pred_resized[mid:]
        return out
