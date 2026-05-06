from __future__ import annotations


class RTPFBError(Exception):
    """Base for all rtpfb-specific errors."""


class ConfigurationError(RTPFBError):
    """Bad / missing config values."""


class ModelNotFoundError(RTPFBError):
    """Required model weights are missing on disk."""


class DependencyMissingError(RTPFBError):
    """An optional Python dependency is not installed."""


class DeviceUnavailableError(RTPFBError):
    """Requested camera / microphone / GPU is not available."""


class VoiceLibraryError(RTPFBError):
    """Generic voice-library error."""


class VoiceModelNotFoundError(VoiceLibraryError):
    """Voice with this name isn't registered."""


class VoiceModelExistsError(VoiceLibraryError):
    """Trying to add a voice that's already registered without --overwrite."""


class StreamingError(RTPFBError):
    """Audio / video streaming pipeline error."""


class VoiceConversionError(RTPFBError):
    """Voice changer (knn-vc / rvc / w-okada) failed."""


class HealthCheckFailed(RTPFBError):
    """Aggregates one or more failed preflight checks."""

    def __init__(self, failures: list[str]):
        self.failures = list(failures)
        super().__init__("preflight failed:\n  - " + "\n  - ".join(self.failures))
