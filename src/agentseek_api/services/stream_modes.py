"""Shared stream-mode normalization and validation.

Used by both the runs API and the cron service so the two paths cannot drift
in which modes they accept. Callers map the raised ``ValueError`` to whatever
HTTP status fits their context (422 for runs, 400 for crons).
"""

from __future__ import annotations

from typing import Any

SUPPORTED_RUN_STREAM_MODES = {
    "values",
    "updates",
    "messages",
    "messages-tuple",
    "debug",
    "events",
    "tasks",
    "checkpoints",
    "custom",
}
RUN_STREAM_MODE_ALIASES: dict[str, str] = {}
DEFAULT_STREAM_MODES = ["values"]


def normalize_stream_modes(stream_mode: Any) -> list[str]:
    """Normalize ``stream_mode`` to a deduped list of supported mode strings.

    ``None`` yields the default ``["values"]``. A bare string is wrapped. An
    empty list, blank entries, or unsupported values raise ``ValueError`` — an
    empty selection is treated as invalid rather than silently meaning "no
    streaming", matching the runs API contract.
    """
    if stream_mode is None:
        return list(DEFAULT_STREAM_MODES)

    raw_modes = [stream_mode] if isinstance(stream_mode, str) else list(stream_mode)
    modes: list[str] = []
    invalid_modes: list[str] = []

    if not raw_modes:
        invalid_modes.append("<empty>")

    for mode in raw_modes:
        raw_mode = str(mode).strip()
        if not raw_mode:
            invalid_modes.append("<empty>")
            continue
        normalized = RUN_STREAM_MODE_ALIASES.get(raw_mode, raw_mode)
        if normalized not in modes:
            modes.append(normalized)

    unsupported = invalid_modes + [mode for mode in modes if mode not in SUPPORTED_RUN_STREAM_MODES]
    if unsupported:
        raise ValueError(
            "Unsupported stream_mode value(s): "
            f"{', '.join(unsupported)}. Supported values: {', '.join(sorted(SUPPORTED_RUN_STREAM_MODES))}."
        )
    return modes
