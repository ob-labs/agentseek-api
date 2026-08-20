from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import AbstractSet

from agentseek_api.environment import ResolvedEnvironment


def resolved_fixture(
    *,
    values: Mapping[str, str],
    declared_keys: AbstractSet[str],
    unresolved_references: Mapping[str, AbstractSet[str]] | None = None,
) -> ResolvedEnvironment:
    return ResolvedEnvironment(
        values=MappingProxyType(dict(values)),
        origins=MappingProxyType({}),
        declared_keys=frozenset(declared_keys),
        unresolved_references=MappingProxyType(
            {
                key: frozenset(names)
                for key, names in (unresolved_references or {}).items()
            }
        ),
        diagnostics=(),
    )
