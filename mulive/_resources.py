"""Paths to read-only files bundled with the Mulive package."""

from importlib.resources import files
from pathlib import Path


def packaged_path(*parts: str) -> Path:
    """Return a filesystem path for a bundled Mulive resource."""
    return Path(files("mulive").joinpath(*parts))
