from __future__ import annotations

from typing import TypedDict

from agentseek_api.core.config_file import get_active_config_payload

DEFAULT_EXPOSE_HEADERS: list[str] = ["Content-Location", "Location"]


class CorsConfig(TypedDict, total=False):
    allow_origins: list[str]
    allow_methods: list[str]
    allow_headers: list[str]
    allow_credentials: bool
    expose_headers: list[str]
    max_age: int


def get_cors_config() -> CorsConfig | None:
    payload = get_active_config_payload()
    if payload is None:
        return None
    http = payload.get("http")
    if not isinstance(http, dict):
        return None
    cors = http.get("cors")
    if not isinstance(cors, dict):
        return None
    return cors  # type: ignore[return-value]
