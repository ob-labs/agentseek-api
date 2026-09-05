from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

EmbeddingFunction = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class StoreTtlConfig:
    refresh_on_read: bool = True
    default_ttl: float | None = None
    sweep_interval_minutes: int | None = None

    def to_runtime_config(self) -> dict[str, Any] | None:
        if (
            self.refresh_on_read is True
            and self.default_ttl is None
            and self.sweep_interval_minutes is None
        ):
            return None
        return {
            "refresh_on_read": self.refresh_on_read,
            "default_ttl": self.default_ttl,
            "sweep_interval_minutes": self.sweep_interval_minutes,
        }


@dataclass(frozen=True)
class StoreIndexConfig:
    embed: EmbeddingFunction | str | None = None
    dims: int | None = None
    fields: list[str] | None = None

    def to_runtime_config(self) -> dict[str, Any] | None:
        if self.embed is None and self.dims is None and self.fields is None:
            return None
        config: dict[str, Any] = {}
        if self.embed is not None:
            config["embed"] = self.embed
        if self.dims is not None:
            config["dims"] = self.dims
        if self.fields is not None:
            config["fields"] = self.fields
        return config


@dataclass(frozen=True)
class StoreConfig:
    ttl: StoreTtlConfig = StoreTtlConfig()
    index: StoreIndexConfig = StoreIndexConfig()


def _active_config_path(agentseek_graphs: str | None) -> Path | None:
    if agentseek_graphs:
        path = Path(agentseek_graphs).expanduser().resolve()
        if path.exists():
            return path
    for candidate in ("agentseek.json", "langgraph.json"):
        path = Path(candidate).resolve()
        if path.exists():
            return path
    return None


def _load_embedding_function(reference: str, *, config_path: Path) -> EmbeddingFunction:
    module_ref, symbol = reference.rsplit(":", maxsplit=1)
    if not module_ref or not symbol:
        raise RuntimeError(f"Invalid store.index.embed reference '{reference}'. Expected 'module:symbol'.")
    if module_ref.endswith(".py") or module_ref.startswith(".") or "/" in module_ref or "\\" in module_ref:
        module_path = Path(module_ref).expanduser()
        if not module_path.is_absolute():
            module_path = config_path.parent / module_path
        module_path = module_path.resolve()
        spec = importlib.util.spec_from_file_location(
            f"agentseek_store_embeddings_{abs(hash(module_path))}",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load store.index.embed module '{module_path}'.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load store.index.embed module '{module_path}' from '{config_path}': {exc}"
            ) from exc
    else:
        try:
            module = importlib.import_module(module_ref)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not import store.index.embed module '{module_ref}' from '{config_path}': {exc}"
            ) from exc
    embed_fn = getattr(module, symbol, None)
    if not callable(embed_fn):
        raise RuntimeError(
            f"Could not resolve callable store.index.embed '{reference}' from '{config_path}'."
        )
    return embed_fn


def _looks_like_python_reference(reference: str) -> bool:
    if ":" not in reference:
        return False
    module_ref, symbol = reference.rsplit(":", maxsplit=1)
    if not module_ref or not symbol:
        return False
    return symbol.isidentifier()


def _apply_config_dependencies(payload: dict[str, object], *, config_path: Path) -> None:
    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, list):
        return
    for dependency in dependencies:
        if not isinstance(dependency, str):
            continue
        if dependency == ".":
            root = config_path.parent.resolve()
        else:
            candidate = Path(dependency).expanduser()
            root = candidate.resolve() if candidate.is_absolute() else (config_path.parent / candidate).resolve()
        if root.exists():
            root_text = str(root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)


def load_store_config(*, agentseek_graphs: str | None) -> StoreConfig:
    config_path = _active_config_path(agentseek_graphs)
    if config_path is None:
        return StoreConfig()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return StoreConfig()
    raw_store = payload.get("store")
    if not isinstance(raw_store, dict):
        return StoreConfig()
    _apply_config_dependencies(payload, config_path=config_path)

    ttl = StoreTtlConfig()
    raw_ttl = raw_store.get("ttl")
    if isinstance(raw_ttl, dict):
        refresh_on_read = raw_ttl.get("refresh_on_read", True)
        default_ttl = raw_ttl.get("default_ttl")
        sweep_interval = raw_ttl.get("sweep_interval_minutes")
        ttl = StoreTtlConfig(
            refresh_on_read=bool(refresh_on_read),
            default_ttl=float(default_ttl) if isinstance(default_ttl, (int, float)) else None,
            sweep_interval_minutes=int(sweep_interval) if isinstance(sweep_interval, int) else None,
        )

    index = StoreIndexConfig()
    raw_index = raw_store.get("index")
    if isinstance(raw_index, dict):
        embed = raw_index.get("embed")
        dims = raw_index.get("dims")
        raw_fields = raw_index.get("fields")
        fields = [item for item in raw_fields if isinstance(item, str)] if isinstance(raw_fields, list) else None
        embed_value: EmbeddingFunction | str | None = None
        if isinstance(embed, str):
            embed_value = _load_embedding_function(embed, config_path=config_path) if _looks_like_python_reference(embed) else embed
        index = StoreIndexConfig(
            embed=embed_value,
            dims=int(dims) if isinstance(dims, int) else None,
            fields=fields,
        )

    return StoreConfig(ttl=ttl, index=index)
