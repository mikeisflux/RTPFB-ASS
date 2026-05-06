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
    target_face_path: str = ""

    # mocap   — mediapipe pose → VMC → external 3D renderer (Unreal+MetaHuman,
    #            Unity, VSeeFace…). Best path for full-body + physics.
    # faceswap — legacy 2D pipeline: InsightFace face swap → virtual cam.
    #            No body, no physics; kept for cases where 3D isn't an option.
    mode: str = "mocap"

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

    # VMC mocap output (used in mode="mocap")
    vmc_host: str = "127.0.0.1"
    vmc_port: int = 39539
    vmc_face_blendshapes: bool = True
    face_landmarker_model: str = "face_landmarker.task"

    audio_samplerate: int = 16000
    audio_channels: int = 1
    audio_blocksize: int = 256

    voice_model: str = ""
    voice_backend: str = "auto"  # "auto" | "rvc" | "knn-vc" | "w-okada"
    voice_ws_url: str = "ws://localhost:18888"
    voice_pitch_shift: float = 0.0
    voice_crossfade_samples: int = 64

    virtual_mic_device: Optional[Union[str, int]] = None
    # Mic input override — useful on Windows where the default input is
    # often a virtual cable instead of the physical mic.
    mic_device: Optional[Union[str, int]] = None

    stabilize_blend: float = 0.35
