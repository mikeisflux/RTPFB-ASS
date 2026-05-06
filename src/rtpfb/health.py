from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ._log import get_logger
from .config import MODELS_DIR, THIRD_PARTY_DIR, PipelineConfig
from .errors import HealthCheckFailed

log = get_logger("rtpfb.health")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class HealthReport:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)
        marker = "ok" if result.ok else "FAIL"
        msg = f"{marker:>4s} {result.name}"
        if result.detail:
            msg += f" — {result.detail}"
        if result.ok:
            log.info(msg)
        else:
            log.warning(msg)

    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def failures(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if not c.ok]

    def raise_if_failed(self) -> None:
        if not self.ok():
            raise HealthCheckFailed(self.failures())


def check_camera(index: int) -> CheckResult:
    try:
        import cv2
    except ImportError:
        return CheckResult("camera", False, "opencv-python not installed")
    cap = cv2.VideoCapture(index)
    is_open = cap.isOpened()
    cap.release()
    if is_open:
        return CheckResult("camera", True, f"index={index}")
    return CheckResult("camera", False, f"could not open camera index {index}")


def check_gpu() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult("gpu", False, "torch not installed (required for voice + lipsync backends)")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        count = torch.cuda.device_count()
        return CheckResult("gpu", True, f"{count} device(s); 0={name}")
    return CheckResult("gpu", False, "no CUDA device — pipeline will run on CPU and be too slow for streaming")


def check_swap_model(name: str) -> CheckResult:
    path = MODELS_DIR / name
    if path.exists():
        return CheckResult(f"swap-model[{name}]", True, str(path))
    return CheckResult(
        f"swap-model[{name}]",
        False,
        f"missing at {path}; see scripts/download_models.sh",
    )


def check_face_image(path: str) -> CheckResult:
    if not path:
        return CheckResult("target-face-image", False, "path not provided")
    p = Path(path)
    if p.exists() and p.is_file():
        return CheckResult("target-face-image", True, str(p))
    return CheckResult("target-face-image", False, f"missing at {p}")


def check_third_party(name: str) -> CheckResult:
    path = THIRD_PARTY_DIR / name
    if path.exists():
        return CheckResult(f"third_party[{name}]", True, str(path))
    return CheckResult(
        f"third_party[{name}]",
        False,
        f"not cloned; run scripts/install_third_party.sh",
    )


def check_pyvirtualcam() -> CheckResult:
    try:
        import pyvirtualcam  # noqa: F401
    except ImportError:
        return CheckResult("virtual-camera-driver", False, "pyvirtualcam not installed")
    return CheckResult("virtual-camera-driver", True, "pyvirtualcam importable")


def check_sounddevice() -> CheckResult:
    try:
        import sounddevice as sd
    except ImportError:
        return CheckResult("sounddevice", False, "sounddevice not installed")
    try:
        devices = sd.query_devices()
        return CheckResult("sounddevice", True, f"{len(devices)} device(s) detected")
    except Exception as exc:  # noqa: BLE001 - surface the underlying message
        return CheckResult("sounddevice", False, f"query_devices failed: {exc}")


def check_voice_in_library(name: str) -> CheckResult:
    if not name:
        return CheckResult("voice-in-library", False, "no voice name provided")
    from .voice_library import VoiceLibrary

    try:
        meta = VoiceLibrary().get(name)
    except KeyError as exc:
        return CheckResult(f"voice[{name}]", False, str(exc))
    return CheckResult(
        f"voice[{name}]",
        True,
        f"method={meta.method} samples={meta.samples_seconds:.1f}s",
    )


def preflight(config: PipelineConfig) -> HealthReport:
    """Run all the checks that apply to ``config`` and return the report.

    Each check is scoped to a feature flag so voice-only / mocap-only
    configs don't fail on irrelevant resources (e.g. camera, swap model,
    target face image).

    Caller decides whether to raise on failure. Use ``report.raise_if_failed()``
    to abort startup, or just log the report and continue if some checks are
    advisory (e.g. GPU on a CPU dev machine).
    """
    report = HealthReport()
    log.info("running preflight…")

    needs_target_face = config.mode == "faceswap" or config.enable_body
    needs_swap_model = config.mode == "faceswap"
    needs_camera = (
        config.enable_pose
        or config.mode == "faceswap"
        or config.enable_lipsync
        or config.output_virtual_camera
    )
    # GPU is required for in-process voice / lipsync backends. The w-okada
    # backend offloads inference to a separate server (its own venv +
    # CUDA), so rtpfb itself doesn't need GPU when using it.
    needs_gpu = (
        config.enable_lipsync
        or (config.enable_voice and config.voice_backend in {"knn-vc", "rvc", "auto"})
    )

    if needs_target_face:
        report.add(check_face_image(config.target_face_path))
    if needs_swap_model:
        report.add(check_swap_model(config.swap_model))
    if needs_camera:
        report.add(check_camera(config.camera_index))

    if needs_gpu:
        report.add(check_gpu())
    if config.enable_voice or config.enable_lipsync:
        report.add(check_sounddevice())

    if config.enable_voice:
        if config.voice_model:
            report.add(check_voice_in_library(config.voice_model))
        if config.voice_backend in {"knn-vc", "auto"}:
            report.add(check_third_party("knn-vc"))
        if config.voice_backend in {"rvc", "auto"}:
            report.add(check_third_party("RVC-WebUI"))

    if config.output_virtual_camera:
        report.add(check_pyvirtualcam())

    if config.mode == "mocap" and config.vmc_face_blendshapes:
        face_model = MODELS_DIR / config.face_landmarker_model
        report.add(
            CheckResult(
                f"face-landmarker[{config.face_landmarker_model}]",
                face_model.exists(),
                str(face_model)
                if face_model.exists()
                else f"missing at {face_model}; will fall back to synthesised blendshapes",
            )
        )

    return report
