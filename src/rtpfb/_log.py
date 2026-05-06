from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional

_CONFIGURED = False


def configure_logging(level: Optional[str] = None, json_output: bool = False) -> None:
    """Initialise the rtpfb logger. Idempotent.

    Level resolution: explicit arg > $RTPFB_LOG_LEVEL > "INFO".
    Set ``RTPFB_LOG_JSON=1`` to emit one JSON object per line (useful when
    shipping logs to a stack-trace-aware aggregator).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    if level is None:
        level = os.environ.get("RTPFB_LOG_LEVEL", "INFO")
    if not json_output:
        json_output = os.environ.get("RTPFB_LOG_JSON", "").lower() in {"1", "true", "yes"}

    root = logging.getLogger("rtpfb")
    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)5s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str = "rtpfb") -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)
