from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, TypeVar
from uuid import UUID

T = TypeVar("T")

DEFAULT_SSE_KEEPALIVE_INTERVAL_SECONDS = 15.0


def _langchain_json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    kwargs.setdefault("default", _langchain_json_default)
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(obj, **kwargs)


def sse_keepalive_comment() -> str:
    return ": keepalive\n\n"


async def iter_with_sse_keepalives(
    source: AsyncIterator[T],
    *,
    interval_seconds: float | None = None,
) -> AsyncIterator[T | None]:
    interval = DEFAULT_SSE_KEEPALIVE_INTERVAL_SECONDS if interval_seconds is None else interval_seconds
    iterator = source.__aiter__()
    pending: asyncio.Task[T] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.create_task(anext(iterator))
            try:
                item = await asyncio.wait_for(asyncio.shield(pending), timeout=interval)
            except TimeoutError:
                yield None
                continue
            except StopAsyncIteration:
                return
            pending = None
            yield item
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            try:
                await pending
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
        aclose = getattr(iterator, "aclose", None)
        if callable(aclose):
            await aclose()
