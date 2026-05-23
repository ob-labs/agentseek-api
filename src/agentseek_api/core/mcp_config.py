from __future__ import annotations

from agentseek_api.core.config_file import get_active_config_payload


def is_mcp_enabled() -> bool:
    payload = get_active_config_payload()
    if payload is None:
        return True
    http = payload.get("http")
    if not isinstance(http, dict):
        return True
    return http.get("disable_mcp") is not True
