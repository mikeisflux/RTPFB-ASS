from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from ._log import configure_logging, get_logger
from .config import PipelineConfig
from .errors import HealthCheckFailed, RTPFBError

def _parse_device_id(raw):
    """sounddevice accepts either an integer index or a name substring.
    A bare numeric CLI string (e.g. ``--mic 18``) means the index, not a
    name to substring-match. Without this conversion the user would have
    to provide a substring unique across MME/DirectSound/WASAPI/WDM-KS."""
    if isinstance(raw, str) and raw.lstrip("-").isdigit():
        return int(raw)
    return raw


_CONFIG_FIELDS_FROM_CLI = {
    "mode": "mode",
    "target": "target_face_path",
    "camera": "camera_index",
    "width": "width",
    "height": "height",
    "fps": "fps",
    "swap_model": "swap_model",
    "no_pose": ("enable_pose", lambda v: not v),
    "voice": "enable_voice",
    "lipsync": "enable_lipsync",
    "body": "enable_body",
    "no_stabilize": ("enable_stabilize", lambda v: not v),
    "no_output": ("output_virtual_camera", lambda v: not v),
    "voice_model": "voice_model",
    "voice_backend": "voice_backend",
    "voice_ws": "voice_ws_url",
    "voice_pitch": "voice_pitch_shift",
    "virtual_mic": ("virtual_mic_device", _parse_device_id),
    "mic_device": ("mic_device", _parse_device_id),
    "audio_blocksize": "audio_blocksize",
    "vmc_host": "vmc_host",
    "vmc_port": "vmc_port",
    "no_vmc_face": ("vmc_face_blendshapes", lambda v: not v),
}


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", help="YAML preset file (CLI flags override preset values)")
    p.add_argument(
        "--mode",
        choices=("mocap", "faceswap"),
        default=None,
        help="mocap = pose → VMC → external 3D renderer; faceswap = legacy 2D path",
    )

    p.add_argument("--camera", type=int, default=None, help="Webcam index")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--fps", type=int, default=None)

    p.add_argument("--no-preflight", action="store_true", help="Skip startup health checks")
    p.add_argument("--no-hud", action="store_true", help="Hide the on-frame telemetry overlay")
    p.add_argument("--log-level", default=None, help="DEBUG / INFO / WARNING / ERROR")

    mocap = p.add_argument_group("mocap mode (mode=mocap)")
    mocap.add_argument("--vmc-host", default=None, dest="vmc_host", help="VMC receiver host (e.g. EVMC4U in Unreal)")
    mocap.add_argument("--vmc-port", type=int, default=None, dest="vmc_port", help="VMC receiver port (default 39539)")
    mocap.add_argument(
        "--no-vmc-face",
        action="store_true",
        default=None,
        help="Disable face blendshape synthesis (pose only)",
    )

    faceswap = p.add_argument_group("faceswap mode (mode=faceswap)")
    faceswap.add_argument("--target", default=None, help="Path to target face image")
    faceswap.add_argument("--swap-model", default=None, dest="swap_model")
    faceswap.add_argument("--no-pose", action="store_true", default=None)
    faceswap.add_argument("--lipsync", action="store_true", default=None, help="Enable Wav2Lip post-process")
    faceswap.add_argument("--body", action="store_true", default=None, help="Enable body re-render stub")
    faceswap.add_argument("--no-stabilize", action="store_true", default=None)
    faceswap.add_argument("--no-output", action="store_true", default=None, help="Don't open a virtual camera")

    voice = p.add_argument_group("voice conversion (target ≤115ms latency)")
    voice.add_argument("--voice", action="store_true", default=None)
    voice.add_argument(
        "--voice-backend",
        choices=("auto", "rvc", "knn-vc", "w-okada"),
        default=None,
        dest="voice_backend",
    )
    voice.add_argument("--voice-model", default=None, dest="voice_model")
    voice.add_argument("--voice-ws", default=None, dest="voice_ws")
    voice.add_argument("--voice-pitch", type=float, default=None, dest="voice_pitch")
    voice.add_argument("--virtual-mic", default=None, dest="virtual_mic")
    voice.add_argument(
        "--mic",
        default=None,
        dest="mic_device",
        help="Input device name (substring match) or index. Override Windows default if it's set to a virtual cable.",
    )
    voice.add_argument("--audio-blocksize", type=int, default=None, dest="audio_blocksize")


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    overrides: dict[str, object] = {}
    for cli_attr, mapping in _CONFIG_FIELDS_FROM_CLI.items():
        raw = getattr(args, cli_attr, None)
        if raw is None:
            continue
        if isinstance(mapping, tuple):
            field, transform = mapping
            overrides[field] = transform(raw)
        else:
            overrides[mapping] = raw

    if args.voice is True or args.virtual_mic is not None:
        overrides["output_virtual_mic"] = True

    if args.preset:
        from .preset import config_from_preset

        return config_from_preset(args.preset, overrides=overrides)

    if overrides.get("mode", "mocap") == "faceswap" and "target_face_path" not in overrides:
        raise SystemExit("--target is required in faceswap mode")
    return PipelineConfig(**overrides)  # type: ignore[arg-type]


def _run_command(args: argparse.Namespace) -> int:
    log = get_logger("rtpfb.cli")
    cfg = _build_config(args)

    if not args.no_preflight:
        from .health import preflight

        report = preflight(cfg)
        if not report.ok():
            log.error("preflight failed:\n  - %s", "\n  - ".join(report.failures()))
            return 2

    from .pipeline import Pipeline

    pipeline = Pipeline(cfg, run_preflight=False, show_hud=not args.no_hud)
    _install_signal_handlers(pipeline)
    pipeline.run()
    return 0


def _health_command(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    from .health import preflight

    report = preflight(cfg)
    return 0 if report.ok() else 2


def _voice_list_command(_args: argparse.Namespace) -> int:
    from .voice_library import VoiceLibrary

    voices = VoiceLibrary().list_voices()
    if not voices:
        print("(no voices registered — use `rtpfb voice add`)")
        return 0
    print(f"{'NAME':20s} {'METHOD':10s} {'SR':>6s} {'SECONDS':>9s}  ACCENT  NOTES")
    for v in voices:
        print(
            f"{v.name:20s} {v.method:10s} {v.sample_rate:6d} {v.samples_seconds:9.1f}  "
            f"{v.accent_tag or '-':6s}  {v.notes}"
        )
    return 0


def _voice_add_command(args: argparse.Namespace) -> int:
    from .voice_library import VoiceLibrary

    meta = VoiceLibrary().add(
        name=args.name,
        samples_dir=Path(args.samples),
        method=args.method,
        notes=args.notes,
        accent_tag=args.accent,
        overwrite=args.overwrite,
    )
    print(f"added voice {meta.name!r} ({meta.method}, {meta.samples_seconds:.1f}s of audio)")
    return 0


def _voice_remove_command(args: argparse.Namespace) -> int:
    from .voice_library import VoiceLibrary

    VoiceLibrary().remove(args.name)
    print(f"removed voice {args.name!r}")
    return 0


def _voice_import_command(args: argparse.Namespace) -> int:
    from .voice_library import VoiceLibrary

    pth = Path(args.pth) if args.pth else None
    index = Path(args.index) if args.index else None
    knnvc_features = Path(args.knnvc_features) if args.knnvc_features else None

    meta = VoiceLibrary().import_voice(
        name=args.name,
        rvc_pth=pth,
        rvc_index=index,
        knnvc_features=knnvc_features,
        accent_tag=args.accent,
        notes=args.notes,
        overwrite=args.overwrite,
    )
    print(f"imported voice {meta.name!r} ({meta.method}, pre-trained)")
    return 0


def _install_signal_handlers(pipeline) -> None:
    log = get_logger("rtpfb.cli")

    def handler(signum, _frame):
        log.info("signal %s received, requesting graceful shutdown", signum)
        pipeline.request_stop()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rtpfb", description="Real-time avatar streaming pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the streaming pipeline")
    _add_run_args(run)
    run.set_defaults(func=_run_command)

    health = sub.add_parser("health", help="Run preflight checks and exit")
    _add_run_args(health)
    health.set_defaults(func=_health_command)

    voice = sub.add_parser("voice", help="Manage cloned voices")
    voice_sub = voice.add_subparsers(dest="voice_command", required=True)

    voice_list = voice_sub.add_parser("list", help="List registered voices")
    voice_list.set_defaults(func=_voice_list_command)

    voice_add = voice_sub.add_parser("add", help="Train / register a voice from sample audio")
    voice_add.add_argument("name", help="Voice name (used with --voice-model)")
    voice_add.add_argument("--samples", required=True, help="Directory of audio samples")
    voice_add.add_argument(
        "--method",
        choices=("knn-vc", "rvc"),
        default="knn-vc",
    )
    voice_add.add_argument("--accent", default="")
    voice_add.add_argument("--notes", default="")
    voice_add.add_argument("--overwrite", action="store_true")
    voice_add.set_defaults(func=_voice_add_command)

    voice_rm = voice_sub.add_parser("remove", help="Remove a voice")
    voice_rm.add_argument("name")
    voice_rm.set_defaults(func=_voice_remove_command)

    voice_imp = voice_sub.add_parser(
        "import",
        help="Import a pre-trained voice model (.pth from RVC, etc.)",
    )
    voice_imp.add_argument("name")
    voice_imp.add_argument("--pth", help="Path to a pre-trained RVC .pth model")
    voice_imp.add_argument("--index", help="Optional matching .index retrieval file")
    voice_imp.add_argument(
        "--knnvc-features",
        dest="knnvc_features",
        help="Path to a pre-computed KNN-VC features.pt (alternative to --pth)",
    )
    voice_imp.add_argument("--accent", default="")
    voice_imp.add_argument("--notes", default="")
    voice_imp.add_argument("--overwrite", action="store_true")
    voice_imp.set_defaults(func=_voice_import_command)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level=getattr(args, "log_level", None))
    log = get_logger("rtpfb.cli")
    try:
        return args.func(args)
    except HealthCheckFailed as exc:
        log.error(str(exc))
        return 2
    except RTPFBError as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        return 1
    except KeyboardInterrupt:
        log.info("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
