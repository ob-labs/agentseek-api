from __future__ import annotations

from agentseek_api.core.config_file import active_config_path, get_active_config_payload


def is_mcp_enabled() -> bool:
    config_path = active_config_path()
    if config_path is None:
        return True
    payload = get_active_config_payload()
    if payload is None:
        return False
    http = payload.get("http")
    if not isinstance(http, dict):
        return True
    return http.get("disable_mcp") is not True
