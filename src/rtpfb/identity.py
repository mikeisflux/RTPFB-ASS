from __future__ import annotations

from typing import Optional

import numpy as np

from .config import MODELS_DIR


class FaceSwapper:
    """Phase 1 MVP: InsightFace inswapper-style face swap.

    Pluggable: this is the place to drop in a DeepFaceLive subprocess,
    a FaceFusion call, or your own ONNX swap engine. The contract is
    ``swap(rgb_frame, pose=...) -> rgb_frame``.
    """

    def __init__(self, target_face_path: str, swap_model: str = "inswapper_128.onnx"):
        import cv2
        import insightface
        from insightface.app import FaceAnalysis

        self._cv2 = cv2

        self._app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=(640, 640))

        model_path = MODELS_DIR / swap_model
        if not model_path.exists():
            raise FileNotFoundError(
                f"Swap model not found at {model_path}. "
                f"See scripts/download_models.sh."
            )
        self._swapper = insightface.model_zoo.get_model(str(model_path))

        target_bgr = cv2.imread(target_face_path)
        if target_bgr is None:
            raise FileNotFoundError(f"Target face image not found: {target_face_path}")
        target_faces = self._app.get(target_bgr)
        if not target_faces:
            raise RuntimeError(f"No face detected in target image {target_face_path}")
        self._target_face = target_faces[0]

    def swap(self, frame_rgb: np.ndarray, pose: Optional[object] = None) -> np.ndarray:
        bgr = self._cv2.cvtColor(frame_rgb, self._cv2.COLOR_RGB2BGR)
        faces = self._app.get(bgr)
        out = bgr
        for face in faces:
            out = self._swapper.get(out, face, self._target_face, paste_back=True)
        return self._cv2.cvtColor(out, self._cv2.COLOR_BGR2RGB)
