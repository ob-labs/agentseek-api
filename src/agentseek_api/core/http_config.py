from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from agentseek_api.core.config_file import active_config_path, get_active_config_payload
from agentseek_api.core.cors_config import CorsConfig


class HttpConfig(TypedDict, total=False):
    app: str
    """Import path for custom FastAPI app, e.g. './custom.py:app' or 'pkg.mod:app'"""
    cors: CorsConfig | None


def get_http_config() -> HttpConfig | None:
    payload = get_active_config_payload()
    if payload is None:
        return None
    http = payload.get("http")
    if not isinstance(http, dict):
        return None
    return http  # type: ignore[return-value]


def get_config_dir() -> Path | None:
    config_path = active_config_path()
    if config_path and config_path.exists():
        return config_path.parent.resolve()
    return None
