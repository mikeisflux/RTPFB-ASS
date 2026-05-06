from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import PipelineConfig
from .pipeline import Pipeline
from .voice_library import METHODS, METHOD_KNNVC, VoiceLibrary


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--target", required=True, help="Path to target face image (used as the persona)")
    p.add_argument("--camera", type=int, default=0, help="Webcam index")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--swap-model", default="inswapper_128.onnx")

    p.add_argument("--no-pose", action="store_true", help="Disable MediaPipe pose tracking")
    p.add_argument("--lipsync", action="store_true", help="Enable Wav2Lip post-process")
    p.add_argument("--body", action="store_true", help="Enable body re-render (Phase 4 stub)")
    p.add_argument("--no-stabilize", action="store_true", help="Disable optical-flow stabilizer")
    p.add_argument("--no-output", action="store_true", help="Don't open a virtual camera; just iterate frames")

    voice = p.add_argument_group("voice conversion (target ≤115ms latency)")
    voice.add_argument("--voice", action="store_true", help="Enable real-time voice-to-voice conversion")
    voice.add_argument(
        "--voice-backend",
        choices=("auto", "rvc", "knn-vc", "w-okada"),
        default="auto",
        help="auto = look up voice in the library; otherwise force a backend",
    )
    voice.add_argument(
        "--voice-model",
        default="",
        help="Voice name from the library (`rtpfb voice list`) — or a w-okada slot id",
    )
    voice.add_argument("--voice-ws", default="ws://localhost:18888", help="WebSocket URL for w-okada backend")
    voice.add_argument("--voice-pitch", type=float, default=0.0, help="Pitch shift in semitones")
    voice.add_argument(
        "--virtual-mic",
        default=None,
        help="sounddevice output device name or index (loopback / VB-CABLE / BlackHole)",
    )
    voice.add_argument(
        "--audio-blocksize",
        type=int,
        default=256,
        help="Samples per audio chunk. 256@16kHz ≈ 16ms; smaller = lower latency, more CPU overhead",
    )


def _run_command(args: argparse.Namespace) -> int:
    cfg = PipelineConfig(
        target_face_path=args.target,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        swap_model=args.swap_model,
        enable_pose=not args.no_pose,
        enable_voice=args.voice,
        enable_lipsync=args.lipsync,
        enable_body=args.body,
        enable_stabilize=not args.no_stabilize,
        output_virtual_camera=not args.no_output,
        output_virtual_mic=args.voice or args.virtual_mic is not None,
        audio_blocksize=args.audio_blocksize,
        voice_model=args.voice_model,
        voice_backend=args.voice_backend,
        voice_ws_url=args.voice_ws,
        voice_pitch_shift=args.voice_pitch,
        virtual_mic_device=args.virtual_mic,
    )
    Pipeline(cfg).run()
    return 0


def _voice_list_command(_args: argparse.Namespace) -> int:
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
    VoiceLibrary().remove(args.name)
    print(f"removed voice {args.name!r}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rtpfb", description="Real-time avatar streaming pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the streaming pipeline")
    _add_run_args(run)
    run.set_defaults(func=_run_command)

    voice = sub.add_parser("voice", help="Manage cloned voices")
    voice_sub = voice.add_subparsers(dest="voice_command", required=True)

    voice_list = voice_sub.add_parser("list", help="List registered voices")
    voice_list.set_defaults(func=_voice_list_command)

    voice_add = voice_sub.add_parser("add", help="Train / register a voice from sample audio")
    voice_add.add_argument("name", help="Voice name (used with --voice-model)")
    voice_add.add_argument(
        "--samples",
        required=True,
        help="Directory of audio samples (.wav/.flac/.mp3/.m4a/.ogg/.opus). 1–5 min recommended for KNN-VC.",
    )
    voice_add.add_argument(
        "--method",
        choices=METHODS,
        default=METHOD_KNNVC,
        help="knn-vc = zero-shot, no training; rvc = full training (highest quality, slow)",
    )
    voice_add.add_argument("--accent", default="", help="Free-form accent tag, e.g. 'en-US-southern'")
    voice_add.add_argument("--notes", default="")
    voice_add.add_argument("--overwrite", action="store_true")
    voice_add.set_defaults(func=_voice_add_command)

    voice_rm = voice_sub.add_parser("remove", help="Remove a voice from the library")
    voice_rm.add_argument("name")
    voice_rm.set_defaults(func=_voice_remove_command)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
