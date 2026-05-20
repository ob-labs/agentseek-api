from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from langgraph.store.base import GetOp, Item, ListNamespacesOp, MatchCondition, PutOp, SearchItem, SearchOp

from agentseek_api.core.runtime_store import UserScopedStore


class FakeBackendStore:
    def __init__(self) -> None:
        self.items: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}
        self.supports_ttl = True
        self.ttl_config = {"default_ttl": 30}
        self.last_batch_ops: list[Any] | None = None

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: bool | list[str] | None = None,
        *,
        ttl: Any,
    ) -> None:
        now = datetime(2026, 5, 19, 12, 0, tzinfo=UTC)
        existing = self.items.get((namespace, key))
        self.items[(namespace, key)] = {
            "value": value,
            "created_at": existing["created_at"] if existing else now,
            "updated_at": now,
            "index": index,
            "ttl": ttl,
        }

    async def aput(self, *args: Any, **kwargs: Any) -> None:
        self.put(*args, **kwargs)

    def get(self, namespace: tuple[str, ...], key: str, *, refresh_ttl: bool | None = None) -> Item | None:
        _ = refresh_ttl
        payload = self.items.get((namespace, key))
        if payload is None:
            return None
        return Item(
            namespace=namespace,
            key=key,
            value=payload["value"],
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
        )

    async def aget(self, *args: Any, **kwargs: Any) -> Item | None:
        return self.get(*args, **kwargs)

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        self.items.pop((namespace, key), None)

    async def adelete(self, *args: Any, **kwargs: Any) -> None:
        self.delete(*args, **kwargs)

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        _ = (query, refresh_ttl)
        matches: list[SearchItem] = []
        for (namespace, key), payload in sorted(self.items.items()):
            if namespace[: len(namespace_prefix)] != namespace_prefix:
                continue
            value = payload["value"]
            if filter and any(value.get(name) != expected for name, expected in filter.items()):
                continue
            matches.append(
                SearchItem(
                    namespace=namespace,
                    key=key,
                    value=value,
                    created_at=payload["created_at"],
                    updated_at=payload["updated_at"],
                    score=None,
                )
            )
        return matches[offset : offset + limit]

    async def asearch(self, *args: Any, **kwargs: Any) -> list[SearchItem]:
        return self.search(*args, **kwargs)

    def list_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        namespaces = sorted({namespace for namespace, _key in self.items})
        if prefix is not None:
            namespaces = [namespace for namespace in namespaces if namespace[: len(prefix)] == prefix]
        if suffix is not None:
            namespaces = [namespace for namespace in namespaces if namespace[-len(suffix) :] == suffix]
        if max_depth is not None:
            namespaces = [namespace[:max_depth] for namespace in namespaces]
        deduped: list[tuple[str, ...]] = []
        for namespace in namespaces:
            if namespace not in deduped:
                deduped.append(namespace)
        return deduped[offset : offset + limit]

    async def alist_namespaces(self, *args: Any, **kwargs: Any) -> list[tuple[str, ...]]:
        return self.list_namespaces(*args, **kwargs)

    def batch(self, ops: list[Any]) -> list[Any]:
        self.last_batch_ops = list(ops)
        results: list[Any] = []
        for op in ops:
            if isinstance(op, PutOp):
                self.put(op.namespace, op.key, op.value or {}, op.index, ttl=op.ttl)
                results.append(None)
            elif isinstance(op, GetOp):
                results.append(self.get(op.namespace, op.key, refresh_ttl=op.refresh_ttl))
            elif isinstance(op, SearchOp):
                results.append(
                    self.search(
                        op.namespace_prefix,
                        query=op.query,
                        filter=op.filter,
                        limit=op.limit,
                        offset=op.offset,
                        refresh_ttl=op.refresh_ttl,
                    )
                )
            elif isinstance(op, ListNamespacesOp):
                prefix = None
                suffix = None
                if op.match_conditions:
                    for condition in op.match_conditions:
                        if condition.match_type == "prefix":
                            prefix = condition.path
                        elif condition.match_type == "suffix":
                            suffix = condition.path
                results.append(
                    self.list_namespaces(
                        prefix=prefix,
                        suffix=suffix,
                        max_depth=op.max_depth,
                        limit=op.limit,
                        offset=op.offset,
                    )
                )
            else:
                raise TypeError(f"Unsupported op type: {type(op)!r}")
        return results

    async def abatch(self, ops: list[Any]) -> list[Any]:
        return self.batch(ops)


def test_user_scoped_store_sync_methods_scope_and_strip_namespaces() -> None:
    backend = FakeBackendStore()
    store = UserScopedStore(backend, user_id="u1")

    store.put(("memories", "users"), "profile", {"kind": "profile"})
    item = store.get(("memories", "users"), "profile", refresh_ttl=False)
    search_results = store.search(("memories",), filter={"kind": "profile"})
    namespaces = store.list_namespaces(prefix=("memories",), max_depth=2)

    assert item is not None
    assert item.namespace == ("memories", "users")
    assert backend.items[(("__agentseek_users__", "u1", "memories", "users"), "profile")]["value"] == {"kind": "profile"}
    assert [result.namespace for result in search_results] == [("memories", "users")]
    assert namespaces == [("memories", "users")]


@pytest.mark.asyncio
async def test_user_scoped_store_async_methods_scope_and_strip_namespaces() -> None:
    backend = FakeBackendStore()
    store = UserScopedStore(backend, user_id="u1")

    await store.aput(("graph", "memory"), "k1", {"kind": "graph"})
    item = await store.aget(("graph", "memory"), "k1", refresh_ttl=False)
    search_results = await store.asearch(("graph",), filter={"kind": "graph"})
    namespaces = await store.alist_namespaces(prefix=("graph",), max_depth=2)
    await store.adelete(("graph", "memory"), "k1")

    assert item is not None
    assert item.namespace == ("graph", "memory")
    assert [result.namespace for result in search_results] == [("graph", "memory")]
    assert namespaces == [("graph", "memory")]
    assert await store.aget(("graph", "memory"), "k1", refresh_ttl=False) is None


def test_user_scoped_store_exposes_backend_ttl_capabilities() -> None:
    store = UserScopedStore(FakeBackendStore(), user_id="u1")

    assert store.supports_ttl is True
    assert store.ttl_config == {"default_ttl": 30}


def test_user_scoped_store_batch_scopes_ops_and_strips_results() -> None:
    backend = FakeBackendStore()
    store = UserScopedStore(backend, user_id="u1")

    results = store.batch(
        [
            PutOp(namespace=("graph", "memory"), key="k1", value={"kind": "graph"}),
            GetOp(namespace=("graph", "memory"), key="k1"),
            SearchOp(namespace_prefix=("graph",), filter={"kind": "graph"}),
            ListNamespacesOp(match_conditions=(MatchCondition(match_type="prefix", path=("graph",)),), max_depth=2),
        ]
    )

    assert isinstance(backend.last_batch_ops, list)
    assert backend.last_batch_ops[0].namespace == ("__agentseek_users__", "u1", "graph", "memory")
    assert backend.last_batch_ops[1].namespace == ("__agentseek_users__", "u1", "graph", "memory")
    assert backend.last_batch_ops[2].namespace_prefix == ("__agentseek_users__", "u1", "graph")
    assert backend.last_batch_ops[3].match_conditions == (
        MatchCondition(match_type="prefix", path=("__agentseek_users__", "u1", "graph")),
    )
    assert results[0] is None
    assert results[1] is not None
    assert results[1].namespace == ("graph", "memory")
    assert [item.namespace for item in results[2]] == [("graph", "memory")]
    assert results[3] == [("graph", "memory")]


def test_user_scoped_store_batch_injects_user_prefix_for_suffix_only_namespace_queries() -> None:
    backend = FakeBackendStore()
    backend.put(("__agentseek_users__", "u1", "graph", "store"), "k1", {"kind": "graph"}, ttl=None)
    backend.put(("__agentseek_users__", "u2", "graph", "store"), "k2", {"kind": "graph"}, ttl=None)
    store = UserScopedStore(backend, user_id="u1")

    results = store.batch(
        [
            ListNamespacesOp(
                match_conditions=(MatchCondition(match_type="suffix", path=("store",)),),
                max_depth=2,
            )
        ]
    )

    assert isinstance(backend.last_batch_ops, list)
    assert backend.last_batch_ops[0].match_conditions == (
        MatchCondition(match_type="prefix", path=("__agentseek_users__", "u1")),
        MatchCondition(match_type="suffix", path=("store",)),
    )
    assert results == [[("graph", "store")]]


@pytest.mark.asyncio
async def test_user_scoped_store_abatch_scopes_ops_and_strips_results() -> None:
    backend = FakeBackendStore()
    store = UserScopedStore(backend, user_id="u1")

    results = await store.abatch(
        [
            PutOp(namespace=("graph", "memory"), key="k1", value={"kind": "graph"}),
            GetOp(namespace=("graph", "memory"), key="k1"),
        ]
    )

    assert isinstance(backend.last_batch_ops, list)
    assert backend.last_batch_ops[0].namespace == ("__agentseek_users__", "u1", "graph", "memory")
    assert backend.last_batch_ops[1].namespace == ("__agentseek_users__", "u1", "graph", "memory")
    assert results[0] is None
    assert results[1] is not None
    assert results[1].namespace == ("graph", "memory")
