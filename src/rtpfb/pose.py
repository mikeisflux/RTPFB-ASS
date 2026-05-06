from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ._log import get_logger
from .config import MODELS_DIR
from .errors import DependencyMissingError, ModelNotFoundError

log = get_logger("rtpfb.pose")


@dataclass
class PoseFrame:
    """Landmarks extracted by MediaPipe Holistic. Coordinates are normalised [0,1]."""

    face_landmarks: Optional[np.ndarray]
    pose_landmarks: Optional[np.ndarray]
    left_hand_landmarks: Optional[np.ndarray]
    right_hand_landmarks: Optional[np.ndarray]


def _to_array(landmark_list) -> Optional[np.ndarray]:
    if landmark_list is None:
        return None
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmark_list.landmark],
        dtype=np.float32,
    )


class PoseTracker:
    """MediaPipe Holistic — face + body + hand landmarks per frame."""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        import mediapipe as mp

        self._mp = mp
        self._holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            refine_face_landmarks=True,
        )

    def process(self, frame_rgb: np.ndarray) -> PoseFrame:
        results = self._holistic.process(frame_rgb)
        return PoseFrame(
            face_landmarks=_to_array(results.face_landmarks),
            pose_landmarks=_to_array(results.pose_landmarks),
            left_hand_landmarks=_to_array(results.left_hand_landmarks),
            right_hand_landmarks=_to_array(results.right_hand_landmarks),
        )

    def close(self) -> None:
        self._holistic.close()


class FaceBlendshapeExtractor:
    """MediaPipe Face Landmarker v2 — outputs all 52 ARKit-standard blendshapes.

    Drops the synthesised 3-blendshape fallback in favour of the model's
    native scores: jawOpen, mouthSmileLeft/Right, eyeBlink*, browInnerUp,
    cheekPuff, tongueOut, etc. Same names ARKit (and therefore MetaHuman /
    EVMC4U / Live Link Face) consumes.

    Usage:
        ext = FaceBlendshapeExtractor()
        bs = ext.extract(frame_rgb)   # dict[str, float]
    """

    def __init__(self, model_path: Optional[str | Path] = None):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise DependencyMissingError(
                "mediapipe (with the tasks API) is required for Face Landmarker v2. "
                "pip install 'mediapipe>=0.10.9'"
            ) from exc

        self._mp = mp

        path = Path(model_path) if model_path else MODELS_DIR / "face_landmarker.task"
        if not path.exists():
            raise ModelNotFoundError(
                f"Face Landmarker model not found at {path}. "
                f"Run scripts/download_models.sh — see the face_landmarker.task entry."
            )

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)
        log.info("Face Landmarker v2 loaded from %s", path)

    def extract(self, frame_rgb: np.ndarray) -> dict[str, float]:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect(mp_image)
        if not result.face_blendshapes:
            return {}
        out: dict[str, float] = {}
        for category in result.face_blendshapes[0]:
            if category.category_name == "_neutral":
                continue
            out[category.category_name] = float(category.score)
        return out

    def close(self) -> None:
        self._detector.close()
