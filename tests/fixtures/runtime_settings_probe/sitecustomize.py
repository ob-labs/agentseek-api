"""Test-only probe loaded by runtime child interpreters."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path


validation_child_pid_path = os.environ.get("AGENTSEEK_VALIDATION_CHILD_PID_PATH")
if validation_child_pid_path:
    Path(validation_child_pid_path).write_text(
        str(os.getpid()),
        encoding="utf-8",
    )


probe_path = os.environ.get("AGENTSEEK_SETTINGS_PROBE_PATH")
if probe_path:
    probe_fields = tuple(
        field
        for field in os.environ["AGENTSEEK_SETTINGS_PROBE_FIELDS"].split(",")
        if field
    )
    probe_exit_code = int(os.environ.get("AGENTSEEK_SETTINGS_PROBE_EXIT_CODE", "0"))

    def _record_settings(awaitable) -> int:
        awaitable.close()
        from agentseek_api.settings import settings

        observed = {
            "pid": os.getpid(),
            "settings": {field: getattr(settings, field) for field in probe_fields},
        }
        Path(probe_path).write_text(
            json.dumps(observed, sort_keys=True),
            encoding="utf-8",
        )
        return probe_exit_code

    asyncio.run = _record_settings
