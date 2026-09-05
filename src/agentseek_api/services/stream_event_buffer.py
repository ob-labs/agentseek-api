from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StreamEvent:
    kind: Literal["run", "thread"]
    stream_id: str
    seq: int
    payload: dict[str, Any]


class StreamEventBuffer:
    """Bound SQL write batches by event count and serialized payload size."""

    def __init__(
        self,
        write: Callable[[list[StreamEvent]], Awaitable[None]],
        *,
        run_id: str,
        thread_id: str,
        max_events: int = 128,
        max_bytes: int = 1024 * 1024,
        flush_interval: float = 0.1,
    ) -> None:
        if max_events <= 0 or max_bytes <= 0 or flush_interval <= 0:
            raise ValueError("Stream buffer limits must be positive")
        self.run_id = run_id
        self.thread_id = thread_id
        self._write = write
        self._max_events = max_events
        self._max_bytes = max_bytes
        self._flush_interval = flush_interval
        self._pending: list[StreamEvent] = []
        self._pending_bytes = 0
        self._lock = asyncio.Lock()
        self._nonempty = asyncio.Event()
        self._closing = asyncio.Event()
        self._closed = False
        self._worker: asyncio.Task[None] | None = None

    async def __aenter__(self) -> StreamEventBuffer:
        self._worker = asyncio.create_task(
            self._flush_periodically(), name="stream-event-flush"
        )
        return self

    async def append(self, record: StreamEvent) -> bool:
        async with self._lock:
            if self._closed:
                return False
            # Producers may reuse nested message dictionaries after publishing.
            snapshot = replace(record, payload=deepcopy(record.payload))
            size = len(json.dumps(snapshot.payload, ensure_ascii=False).encode("utf-8"))
            if self._pending and self._pending_bytes + size > self._max_bytes:
                await self._flush_locked()
            self._pending.append(snapshot)
            self._pending_bytes += size
            self._nonempty.set()
            if (
                len(self._pending) >= self._max_events
                or self._pending_bytes >= self._max_bytes
            ):
                # Holding the lock applies backpressure while storage is slow.
                # A single oversized event is written alone, never accumulated.
                await self._flush_locked()
            return True

    async def _flush_locked(self) -> None:
        if not self._pending:
            return
        # Retain the batch until the write finishes; cancellation can interrupt
        # an in-flight transaction, and closing the buffer must retry it.
        await self._write(self._pending)
        self._pending = []
        self._pending_bytes = 0
        self._nonempty.clear()

    async def _flush_periodically(self) -> None:
        while not self._closed:
            await self._nonempty.wait()
            try:
                await asyncio.wait_for(
                    self._closing.wait(), timeout=self._flush_interval
                )
            except TimeoutError:
                pass
            async with self._lock:
                await self._flush_locked()

    async def _finish(self) -> None:
        self._closing.set()
        self._nonempty.set()
        if self._worker is not None:
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception(
                    "Periodic stream persistence failed; retrying on close"
                )
        async with self._lock:
            await self._flush_locked()

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        self._closed = True
        finish = asyncio.create_task(self._finish(), name="stream-event-drain")
        cancelled = False
        while not finish.done():
            try:
                await asyncio.shield(finish)
            except asyncio.CancelledError:
                cancelled = True
        finish.result()
        if cancelled:
            raise asyncio.CancelledError
