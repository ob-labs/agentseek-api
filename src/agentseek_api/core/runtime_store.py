from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

from langchain_oceanbase.store import OceanBaseStore
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    NOT_PROVIDED,
    PutOp,
    SearchItem,
    SearchOp,
)
from sqlalchemy import MetaData, create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.pool import StaticPool

_USER_NAMESPACE_ROOT = "__agentseek_users__"


def _strip_async_driver(url: str) -> str:
    parsed = make_url(url)
    if parsed.drivername.startswith("sqlite+"):
        return parsed.set(drivername="sqlite").render_as_string(hide_password=False)
    return parsed.render_as_string(hide_password=False)


class SqliteStore(OceanBaseStore):
    def __init__(
        self,
        *,
        url: str,
        index: dict[str, Any] | None = None,
        ttl_config: dict[str, Any] | None = None,
        table_name: str = "langgraph_store_items",
    ) -> None:
        self._sqlite_url = _strip_async_driver(url)
        super().__init__(
            connection_args={
                "host": "localhost",
                "port": "0",
                "user": "",
                "password": "",
                "db_name": self._sqlite_url,
            },
            index=index,
            ttl_config=ttl_config,
            table_name=table_name,
        )

    def _create_client(self, **_kwargs: Any) -> None:
        parsed = make_url(self._sqlite_url)
        engine_kwargs: dict[str, Any] = {"connect_args": {"check_same_thread": False}}
        if parsed.database in {None, "", ":memory:"}:
            engine_kwargs["poolclass"] = StaticPool
        self.obvector = SimpleNamespace(
            engine=create_engine(self._sqlite_url, **engine_kwargs),
            metadata_obj=MetaData(),
        )


def make_user_store_namespace(*, user_id: str, namespace: tuple[str, ...]) -> tuple[str, ...]:
    return (_USER_NAMESPACE_ROOT, user_id, *namespace)


class UserScopedStore:
    def __init__(self, store: BaseStore, *, user_id: str) -> None:
        self._store = store
        self._user_prefix = (_USER_NAMESPACE_ROOT, user_id)

    @property
    def supports_ttl(self) -> bool:
        return bool(getattr(self._store, "supports_ttl", False))

    @property
    def ttl_config(self) -> Any:
        return getattr(self._store, "ttl_config", None)

    def _scope_namespace(self, namespace: tuple[str, ...]) -> tuple[str, ...]:
        return (*self._user_prefix, *namespace)

    def _strip_namespace(self, namespace: tuple[str, ...]) -> tuple[str, ...]:
        if namespace[: len(self._user_prefix)] != self._user_prefix:
            raise ValueError("Store item namespace does not match the authenticated user scope.")
        return namespace[len(self._user_prefix) :]

    def _strip_item(self, item: Item | None) -> Item | None:
        if item is None:
            return None
        return Item(
            namespace=self._strip_namespace(item.namespace),
            key=item.key,
            value=item.value,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )

    def _strip_search_item(self, item: SearchItem) -> SearchItem:
        return SearchItem(
            namespace=self._strip_namespace(item.namespace),
            key=item.key,
            value=item.value,
            created_at=item.created_at,
            updated_at=item.updated_at,
            score=item.score,
        )

    def _scope_match_condition(self, condition: MatchCondition) -> MatchCondition:
        if condition.match_type == "prefix":
            return MatchCondition(match_type="prefix", path=self._scope_namespace(condition.path))
        return condition

    def _scope_op(self, op: Any) -> Any:
        if isinstance(op, PutOp):
            return PutOp(
                namespace=self._scope_namespace(op.namespace),
                key=op.key,
                value=op.value,
                index=op.index,
                ttl=op.ttl,
            )
        if isinstance(op, GetOp):
            return GetOp(
                namespace=self._scope_namespace(op.namespace),
                key=op.key,
                refresh_ttl=op.refresh_ttl,
            )
        if isinstance(op, SearchOp):
            return SearchOp(
                namespace_prefix=self._scope_namespace(op.namespace_prefix),
                filter=op.filter,
                limit=op.limit,
                offset=op.offset,
                query=op.query,
                refresh_ttl=op.refresh_ttl,
            )
        if isinstance(op, ListNamespacesOp):
            conditions = None
            if op.match_conditions is not None:
                conditions = tuple(self._scope_match_condition(condition) for condition in op.match_conditions)
            max_depth = None if op.max_depth is None else len(self._user_prefix) + op.max_depth
            return ListNamespacesOp(
                match_conditions=conditions,
                max_depth=max_depth,
                limit=op.limit,
                offset=op.offset,
            )
        return op

    def _strip_batch_result(self, result: Any) -> Any:
        if isinstance(result, Item):
            return self._strip_item(result)
        if isinstance(result, SearchItem):
            return self._strip_search_item(result)
        if isinstance(result, list):
            stripped: list[Any] = []
            for item in result:
                if isinstance(item, tuple):
                    stripped.append(self._strip_namespace(item))
                elif isinstance(item, SearchItem):
                    stripped.append(self._strip_search_item(item))
                elif isinstance(item, Item):
                    stripped.append(self._strip_item(item))
                else:
                    stripped.append(item)
            return stripped
        return result

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: bool | list[str] | None = None,
        *,
        ttl: Any = NOT_PROVIDED,
    ) -> None:
        self._store.put(self._scope_namespace(namespace), key, value, index=index, ttl=ttl)

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        return self._strip_item(self._store.get(self._scope_namespace(namespace), key, refresh_ttl=refresh_ttl))

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        self._store.delete(self._scope_namespace(namespace), key)

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
        results = self._store.search(
            self._scope_namespace(namespace_prefix),
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
            refresh_ttl=refresh_ttl,
        )
        return [self._strip_search_item(item) for item in results]

    def list_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        scoped_prefix = self._scope_namespace(prefix or ())
        scoped_max_depth = None if max_depth is None else len(self._user_prefix) + max_depth
        namespaces = self._store.list_namespaces(
            prefix=scoped_prefix,
            suffix=suffix,
            max_depth=scoped_max_depth,
            limit=limit,
            offset=offset,
        )
        return [self._strip_namespace(namespace) for namespace in namespaces]

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        scoped_ops = [self._scope_op(op) for op in ops]
        results = self._store.batch(scoped_ops)
        return [self._strip_batch_result(result) for result in results]

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        *,
        index: bool | list[str] | None = None,
        ttl: Any = NOT_PROVIDED,
    ) -> None:
        await self._store.aput(self._scope_namespace(namespace), key, value, index=index, ttl=ttl)

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        return self._strip_item(await self._store.aget(self._scope_namespace(namespace), key, refresh_ttl=refresh_ttl))

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        await self._store.adelete(self._scope_namespace(namespace), key)

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        results = await self._store.asearch(
            self._scope_namespace(namespace_prefix),
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
            refresh_ttl=refresh_ttl,
        )
        return [self._strip_search_item(item) for item in results]

    async def alist_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        scoped_prefix = self._scope_namespace(prefix or ())
        scoped_max_depth = None if max_depth is None else len(self._user_prefix) + max_depth
        namespaces = await self._store.alist_namespaces(
            prefix=scoped_prefix,
            suffix=suffix,
            max_depth=scoped_max_depth,
            limit=limit,
            offset=offset,
        )
        return [self._strip_namespace(namespace) for namespace in namespaces]

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        scoped_ops = [self._scope_op(op) for op in ops]
        results = await self._store.abatch(scoped_ops)
        return [self._strip_batch_result(result) for result in results]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


__all__ = ["SqliteStore", "UserScopedStore", "make_user_store_namespace"]
