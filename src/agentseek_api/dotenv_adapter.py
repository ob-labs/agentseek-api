"""Strict adapter around the supported python-dotenv implementation APIs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dotenv.parser import parse_stream
from dotenv.variables import parse_variables


class DotenvFileError(ValueError):
    def __init__(
        self,
        path: Path,
        message: str,
        *,
        line: int | None = None,
    ) -> None:
        self.path = path
        self.line = line
        location = f" at line {line}" if line is not None else ""
        super().__init__(f"Env file '{path}' {message}{location}.")


def _resolve_value(
    value: str,
    *,
    context: Mapping[str, str | None],
) -> str:
    return "".join(atom.resolve(context) for atom in parse_variables(value))


def parse_dotenv_file(
    path: Path,
    *,
    ambient: Mapping[str, str],
) -> dict[str, str | None]:
    try:
        with path.open(encoding="utf-8") as stream:
            bindings = list(parse_stream(stream))
    except FileNotFoundError as exc:
        raise DotenvFileError(path, "does not exist") from exc
    except UnicodeDecodeError as exc:
        raise DotenvFileError(path, "is not valid UTF-8") from exc
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise DotenvFileError(path, f"could not be read: {reason}") from exc

    malformed = next((binding for binding in bindings if binding.error), None)
    if malformed is not None:
        raise DotenvFileError(
            path,
            "has malformed dotenv syntax",
            line=malformed.original.line,
        )

    context: dict[str, str | None] = dict(ambient)
    values: dict[str, str | None] = {}
    for binding in bindings:
        if binding.key is None:
            continue
        if binding.value is None:
            value = None
        else:
            value = _resolve_value(binding.value, context=context)
        values[binding.key] = value
        context[binding.key] = value
    return values
