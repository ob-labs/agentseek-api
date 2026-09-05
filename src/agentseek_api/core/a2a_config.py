from __future__ import annotations

from agentseek_api.core.config_file import active_config_path, get_active_config_payload


def is_a2a_enabled() -> bool:
    config_path = active_config_path()
    if config_path is None:
        return True
    payload = get_active_config_payload()
    if payload is None:
        return False
    if "http" not in payload:
        return True
    http = payload.get("http")
    if not isinstance(http, dict):
        return False
    disable_a2a = http.get("disable_a2a")
    if disable_a2a is None:
        return True
    if isinstance(disable_a2a, bool):
        return disable_a2a is not True
    return False
