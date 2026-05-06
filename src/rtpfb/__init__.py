"""rtpfb — Real-Time Photorealistic Full-Body Avatar Streaming.

Public API. Each name is lazy-loaded so importing the package doesn't pay
for heavy ML dependencies (mediapipe, insightface, torch) until you actually
construct the relevant class.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "PipelineConfig",
    "Pipeline",
    "VoiceLibrary",
    "VoiceMeta",
    "preflight",
    "configure_logging",
    "RTPFBError",
    "__version__",
]


def __getattr__(name: str):  # PEP 562 lazy attribute access
    if name == "PipelineConfig":
        from .config import PipelineConfig

        return PipelineConfig
    if name == "Pipeline":
        from .pipeline import Pipeline

        return Pipeline
    if name == "VoiceLibrary":
        from .voice_library import VoiceLibrary

        return VoiceLibrary
    if name == "VoiceMeta":
        from .voice_library import VoiceMeta

        return VoiceMeta
    if name == "preflight":
        from .health import preflight

        return preflight
    if name == "configure_logging":
        from ._log import configure_logging

        return configure_logging
    if name == "RTPFBError":
        from .errors import RTPFBError

        return RTPFBError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
