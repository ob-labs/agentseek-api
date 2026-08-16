from __future__ import annotations

import importlib
import runpy
import sys
from collections.abc import Sequence

from pydantic import ValidationError


TARGET_MODULES = {
    "uvicorn": "uvicorn.__main__",
    "worker": "agentseek_api.worker",
    "scheduler": "agentseek_api.scheduler",
}


def _format_settings_validation_error(exc: ValidationError) -> str:
    fields = sorted(
        {
            ".".join(str(part) for part in error["loc"]) + f" ({error['type']})"
            for error in exc.errors(include_input=False, include_url=False)
        }
    )
    return f"Invalid runtime setting(s): {', '.join(fields)}."


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in TARGET_MODULES:
        sys.stderr.write("Invalid internal runtime target.\n")
        return 2
    target_name, *target_argv = arguments
    if target_argv[:1] == ["--"]:
        target_argv = target_argv[1:]
    target_module = TARGET_MODULES[target_name]
    previous_argv = sys.argv
    sys.argv = [target_module, *target_argv]
    try:
        try:
            importlib.import_module("agentseek_api.settings")
        except ValidationError as exc:
            sys.stderr.write(_format_settings_validation_error(exc) + "\n")
            return 2
        try:
            runpy.run_module(target_module, run_name="__main__")
        except SystemExit as exc:
            return (
                exc.code
                if isinstance(exc.code, int)
                else (0 if exc.code is None else 1)
            )
        return 0
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
