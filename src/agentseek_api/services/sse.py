from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TypeVar

T = TypeVar("T")

DEFAULT_SSE_KEEPALIVE_INTERVAL_SECONDS = 15.0


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
