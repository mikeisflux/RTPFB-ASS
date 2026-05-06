from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PoseFrame:
    """Landmarks extracted by MediaPipe Holistic. Coordinates are normalized [0,1]."""

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
