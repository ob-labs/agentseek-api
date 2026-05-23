from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentseek_api.settings import settings


def active_config_path() -> Path | None:
    if settings.AGENTSEEK_GRAPHS:
        path = Path(settings.AGENTSEEK_GRAPHS).expanduser().resolve()
        if path.exists():
            return path
    for candidate in ("agentseek.json", "langgraph.json"):
        path = Path(candidate).resolve()
        if path.exists():
            return path
    return None


def get_active_config_payload() -> dict[str, Any] | None:
    config_path = active_config_path()
    if config_path is None:
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None
