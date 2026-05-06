from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

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
    enable_voice: bool = False
    enable_lipsync: bool = False
    enable_body: bool = False
    enable_stabilize: bool = True
    output_virtual_camera: bool = True
    output_virtual_mic: bool = False

    audio_samplerate: int = 16000
    audio_channels: int = 1
    # 256 samples @ 16kHz ≈ 16ms. Budget for 115ms total: keep this small.
    audio_blocksize: int = 256

    # Voice conversion
    voice_model: str = ""
    voice_backend: str = "w-okada"  # "rvc" | "w-okada"
    voice_ws_url: str = "ws://localhost:18888"
    voice_pitch_shift: float = 0.0
    voice_crossfade_samples: int = 64

    virtual_mic_device: Optional[Union[str, int]] = None

    stabilize_blend: float = 0.35
