#!/usr/bin/env python3
"""Print a bounded, value-free tail from a hosted container smoke log."""

from __future__ import annotations

import re
import sys
from pathlib import Path


MAXIMUM_TAIL_BYTES = 12_000
_REDACTION = b"<redacted>"
_URL_USERINFO = re.compile(rb"(?i)(https?://)[^/\s:@]+:[^@\s/]+@")


def value_free_log_tail(log_path: Path, environment_path: Path) -> str:
    output = log_path.read_bytes()[-MAXIMUM_TAIL_BYTES:]
    for line in environment_path.read_bytes().splitlines():
        _name, separator, value = line.partition(b"=")
        if separator and value:
            output = output.replace(value, _REDACTION)
    output = _URL_USERINFO.sub(rb"\1<redacted>@", output)
    output = output[-MAXIMUM_TAIL_BYTES:]
    return output.decode("utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        print("application startup diagnostic unavailable", file=sys.stderr)
        return 2
    try:
        diagnostic = value_free_log_tail(
            Path(arguments[0]),
            Path(arguments[1]),
        )
    except OSError:
        print("application startup diagnostic unavailable", file=sys.stderr)
        return 1
    if diagnostic:
        print(diagnostic, end="" if diagnostic.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
