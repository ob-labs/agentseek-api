"""Strict adapter around the supported python-dotenv implementation APIs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Iterator, Sequence

from dotenv.parser import parse_stream
from dotenv.variables import Variable, parse_variables


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


@dataclass(frozen=True)
class DotenvBinding:
    """One strict dotenv binding, retaining source order but not its value in repr."""

    key: str | None
    atoms: tuple[object, ...] = field(repr=False)
    line: int = 0
    referenced_names: frozenset[str] = frozenset()
    has_value: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(self, "referenced_names", frozenset(self.referenced_names))


@dataclass(frozen=True)
class DotenvDocument(Sequence[DotenvBinding]):
    """An immutable, validated dotenv source document."""

    path: Path
    bindings: tuple[DotenvBinding, ...] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", tuple(self.bindings))

    def __len__(self) -> int:
        return len(self.bindings)

    def __getitem__(
        self, index: int | slice
    ) -> DotenvBinding | tuple[DotenvBinding, ...]:
        return self.bindings[index]

    def __iter__(self) -> Iterator[DotenvBinding]:
        return iter(self.bindings)


def parse_dotenv_document(path: Path) -> DotenvDocument:
    """Read a dotenv file once and return strict physical-order bindings.

    This adapter deliberately keeps python-dotenv's supported parser and
    interpolation atoms behind one boundary.  The atoms retain enough
    information for target-specific resolution while dataclass repr output
    stays value-free.
    """

    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DotenvFileError(path, "does not exist") from exc
    except UnicodeDecodeError as exc:
        raise DotenvFileError(path, "is not valid UTF-8") from exc
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise DotenvFileError(path, f"could not be read: {reason}") from exc

    bindings = list(parse_stream(StringIO(contents)))
    malformed = next((binding for binding in bindings if binding.error), None)
    if malformed is not None:
        raise DotenvFileError(
            path,
            "has malformed dotenv syntax",
            line=malformed.original.line,
        )

    document_bindings: list[DotenvBinding] = []
    for binding in bindings:
        atoms = (
            tuple(parse_variables(binding.value)) if binding.value is not None else ()
        )
        references = frozenset(
            atom.name for atom in atoms if isinstance(atom, Variable)
        )
        document_bindings.append(
            DotenvBinding(
                key=binding.key,
                atoms=atoms,
                line=binding.original.line,
                referenced_names=references,
                has_value=binding.value is not None,
            )
        )
    return DotenvDocument(path=path, bindings=tuple(document_bindings))


def resolve_dotenv_document(
    document: DotenvDocument,
    *,
    ambient: Mapping[str, str | None],
) -> dict[str, str | None]:
    """Resolve a validated document with python-dotenv's file-local ordering."""

    context: dict[str, str | None] = dict(ambient)
    values: dict[str, str | None] = {}
    for binding in document:
        if binding.key is None:
            continue
        value = (
            "".join(atom.resolve(context) for atom in binding.atoms)
            if binding.has_value
            else None
        )
        values[binding.key] = value
        context[binding.key] = value
    return values


def parse_dotenv_file(
    path: Path,
    *,
    ambient: Mapping[str, str],
) -> dict[str, str | None]:
    return resolve_dotenv_document(parse_dotenv_document(path), ambient=ambient)
