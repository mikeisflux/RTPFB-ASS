from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
ASSETS_DIR = REPO_ROOT / "assets"


@dataclass
class PipelineConfig:
    target_face_path: str

    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30

    swap_model: str = "inswapper_128.onnx"

    enable_pose: bool = True
    enable_lipsync: bool = False
    enable_body: bool = False
    enable_stabilize: bool = True
    output_virtual_camera: bool = True

    audio_samplerate: int = 16000
    audio_channels: int = 1
    audio_blocksize: int = 1024

    stabilize_blend: float = 0.35
