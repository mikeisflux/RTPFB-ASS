from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Optional

from .config import PipelineConfig
from .errors import ConfigurationError, DependencyMissingError


def load_preset(path: str | Path) -> dict[str, Any]:
    """Read a YAML preset file into a plain dict."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DependencyMissingError(
            "PyYAML is required for presets. pip install pyyaml"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Preset not found: {p}")
    data = yaml.safe_load(p.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"Preset must be a YAML mapping at top level, got {type(data).__name__}")
    return data


def config_from_preset(
    path: str | Path,
    overrides: Optional[dict[str, Any]] = None,
) -> PipelineConfig:
    """Build a ``PipelineConfig`` from a preset, with optional overrides.

    ``overrides`` win over preset values. None-valued overrides are dropped
    so CLI args that the user didn't supply don't blow away preset values.
    """
    data = load_preset(path)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                data[k] = v
    return _instantiate_config(data)


def _instantiate_config(data: dict[str, Any]) -> PipelineConfig:
    valid = {f.name for f in fields(PipelineConfig)}
    unknown = set(data) - valid
    if unknown:
        raise ConfigurationError(
            f"Unknown preset keys: {sorted(unknown)}. Valid: {sorted(valid)}"
        )
    if "target_face_path" not in data or not data["target_face_path"]:
        raise ConfigurationError(
            "target_face_path is required (set it in the preset or pass --target on the CLI)"
        )
    return PipelineConfig(**data)
