import asyncio
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from typing import Any


class RunEventBroker:
    def __init__(self, *, max_completed_runs: int = 1024) -> None:
        self._events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._signals: dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self._completed_runs: set[str] = set()
        self._completed_order: deque[str] = deque()
        self._max_completed_runs = max_completed_runs

    def publish(self, run_id: str, event: str, **payload: Any) -> None:
        event_payload = {"event": event, **payload}
        self._events[run_id].append(event_payload)
        if event == "start":
            self._completed_runs.discard(run_id)
            try:
                self._completed_order.remove(run_id)
            except ValueError:
                pass
        if event == "end" and run_id not in self._completed_runs:
            self._completed_runs.add(run_id)
            self._completed_order.append(run_id)
            self._prune_completed_runs()
        self._signals[run_id].set()

    def snapshot(self, run_id: str) -> list[dict[str, Any]]:
        return [dict(event) for event in self._events.get(run_id, [])]

    def _prune_completed_runs(self) -> None:
        while len(self._completed_order) > self._max_completed_runs:
            stale_run_id = self._completed_order.popleft()
            self._completed_runs.discard(stale_run_id)
            self._events.pop(stale_run_id, None)
            self._signals.pop(stale_run_id, None)

    async def stream(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        seen = 0
        while True:
            events = self._events.get(run_id, [])
            while seen < len(events):
                event = dict(events[seen])
                seen += 1
                yield event
            if run_id in self._completed_runs:
                return
            signal = self._signals[run_id]
            signal.clear()
            await signal.wait()


run_broker = RunEventBroker()
