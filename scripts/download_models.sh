#!/usr/bin/env bash
# Pointers for downloading the model weights this pipeline needs.
# We do NOT auto-download non-commercial weights — you accept their terms by
# fetching them. URLs change frequently; see THIRD_PARTY.md for license notes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
mkdir -p "$MODELS_DIR"

cat <<EOF
Place the following files in: $MODELS_DIR

  inswapper_128.onnx       — face swap (Phase 1)
                             Source: InsightFace / Picsi.AI release
                             License: NON-COMMERCIAL

  buffalo_l/               — InsightFace detection + recognition pack
                             Auto-downloaded by insightface on first run, or
                             grab from https://github.com/deepinsight/insightface/releases

  wav2lip_gan.pth          — lip sync (Phase 2)
                             Source: Rudrabha/Wav2Lip releases
                             License: research / non-commercial

  AnimateAnyone/           — Phase 4 weights (optional, large)
                             See third_party/AnimateAnyone/README.md

  face_landmarker.task     — MediaPipe Face Landmarker v2 (mocap mode)
                             52 ARKit-standard blendshapes natively.
                             License: Apache-2.0.
                             Auto-fetch:
                               curl -L -o "$MODELS_DIR/face_landmarker.task" \\
                                 https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

By placing these files in the models/ directory you affirm you have rights to
their use under each model's license.
EOF

# Auto-fetch the Apache-licensed Face Landmarker model since it's required
# for the recommended mocap path.
FACE_TASK="$MODELS_DIR/face_landmarker.task"
if [ ! -f "$FACE_TASK" ]; then
    echo
    echo "Fetching face_landmarker.task (MediaPipe Face Landmarker v2, Apache-2.0)…"
    curl -L --fail -o "$FACE_TASK" \
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
        || echo "  (skipped — network unavailable; download manually later)"
fi
