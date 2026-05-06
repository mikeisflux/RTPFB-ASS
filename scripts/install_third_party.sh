#!/usr/bin/env bash
# Clone the upstream repos referenced by the whitepaper into ./third_party/.
# We do NOT vendor their source — third_party/ is git-ignored. See THIRD_PARTY.md
# for the licenses you inherit by linking to each.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_ROOT/third_party"
mkdir -p "$DEST"
cd "$DEST"

clone_if_missing() {
    local url="$1"
    local dir="$2"
    if [ -d "$dir/.git" ]; then
        echo "[skip] $dir already cloned"
    else
        echo "[clone] $url -> $dir"
        git clone --depth 1 "$url" "$dir"
    fi
}

clone_if_missing https://github.com/iperov/DeepFaceLive.git           DeepFaceLive
clone_if_missing https://github.com/facefusion/facefusion.git         facefusion
clone_if_missing https://github.com/Rudrabha/Wav2Lip.git              Wav2Lip
clone_if_missing https://github.com/OpenTalker/SadTalker.git          SadTalker
clone_if_missing https://github.com/HumanAIGC/AnimateAnyone.git       AnimateAnyone
clone_if_missing https://github.com/magic-research/magicanimate.git   magicanimate
clone_if_missing https://github.com/KwaiVGI/LivePortrait.git          LivePortrait
clone_if_missing https://github.com/myshell-ai/OpenVoice.git          OpenVoice
clone_if_missing https://github.com/Ramya646/ai-avatar-generator.git  ai-avatar-generator

# Optional / heavy — uncomment if you want the 3D Gaussian path.
# clone_if_missing https://github.com/graphdeco-inria/gaussian-splatting.git gaussian-splatting

echo
echo "Done. Upstream repos are in $DEST"
echo "Each retains its own license — see THIRD_PARTY.md."
