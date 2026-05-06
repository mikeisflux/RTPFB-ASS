# Real-time avatar streaming — CUDA 12 runtime + Python 3.11.
# Run with --gpus all for GPU passthrough; mount your camera/audio devices via
# --device /dev/video0 + --device /dev/snd or PulseAudio socket.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RTPFB_LOG_LEVEL=INFO

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    git ffmpeg \
    libsm6 libxext6 libgl1 libglib2.0-0 libsndfile1 \
    pulseaudio v4l2loopback-utils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /app

COPY pyproject.toml requirements.txt README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY presets/ ./presets/

RUN python -m pip install --upgrade pip && \
    pip install -e ".[lipsync,voice,preset]"

# Place model weights / cloned upstreams via volume mounts at runtime to
# keep the image small. Recommended:
#   -v $(pwd)/models:/app/models
#   -v $(pwd)/third_party:/app/third_party
#   -v $(pwd)/voices:/app/voices

ENTRYPOINT ["rtpfb"]
CMD ["--help"]
