from __future__ import annotations

from typing import TypedDict


DEFAULT_EXPOSE_HEADERS: list[str] = ["Content-Location", "Location"]


class CorsConfig(TypedDict, total=False):
    allow_origins: list[str]
    allow_origin_regex: str | None
    allow_methods: list[str]
    allow_headers: list[str]
    allow_credentials: bool
    expose_headers: list[str]
    max_age: int


def get_cors_config() -> CorsConfig | None:
    from agentseek_api.core.http_config import get_http_config

    http_config = get_http_config()
    if not http_config:
        return None
    cors = http_config.get("cors")
    if not isinstance(cors, dict):
        return None
    return cors  # type: ignore[return-value]
