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
    p.add_argument("--lipsync", action="store_true", help="Enable Wav2Lip post-process (Phase 2 stub)")
    p.add_argument("--body", action="store_true", help="Enable body re-render (Phase 4 stub)")
    p.add_argument("--no-stabilize", action="store_true", help="Disable optical-flow stabilizer")
    p.add_argument("--no-output", action="store_true", help="Don't open a virtual camera; just iterate frames")

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
        enable_lipsync=args.lipsync,
        enable_body=args.body,
        enable_stabilize=not args.no_stabilize,
        output_virtual_camera=not args.no_output,
    )
    Pipeline(cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
