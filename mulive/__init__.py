"""Public Mulive API."""

from typing import Any

__all__ = ["mount_voice"]


def mount_voice(*args: Any, **kwargs: Any):
    """Mount Mulive voice handling into a FastAPI application.

    FastAPI remains an optional dependency until this API is used.
    """
    from .server.server_fastapi_embed import mount_voice as _mount_voice

    return _mount_voice(*args, **kwargs)
