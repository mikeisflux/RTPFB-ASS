from __future__ import annotations

import argparse
import sys

from .config import PipelineConfig
from .pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rtpfb", description="Real-time avatar streaming pipeline")
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
        choices=("rvc", "w-okada"),
        default="w-okada",
        help="In-process RVC, or talk to a running w-okada/voice-changer server",
    )
    voice.add_argument("--voice-model", default="", help="Voice model name (RVC slot or w-okada model id)")
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

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

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


if __name__ == "__main__":
    sys.exit(main())
