"""Base path and configuration readers, without loading .env at import time."""
from __future__ import annotations

import os
from pathlib import Path

# The package is one level below the root; the working directory does not affect paths.
BASE = Path(__file__).resolve().parent.parent


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean option; treat true, 1, yes, y and on as true."""
    return os.getenv(name, str(default)).strip().lower() in {"true", "1", "yes", "y", "on"}


def env_int(name: str, default: int, low: int, high: int) -> int:
    """Read an environment integer and reject values outside inclusive bounds."""
    n = int(os.getenv(name, default))
    if not low <= n <= high:
        raise ValueError(f"{name} must be between {low} and {high} (got {n})")
    return n
