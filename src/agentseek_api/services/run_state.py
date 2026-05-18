import asyncio
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from typing import Any


class RunEventBroker:
    def __init__(self, *, max_completed_runs: int = 1024) -> None:
        self._events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._seqs: dict[str, list[int]] = defaultdict(list)
        self._next_seq: dict[str, int] = defaultdict(lambda: 1)
        self._signals: dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self._completed_runs: set[str] = set()
        self._completed_order: deque[str] = deque()
        self._max_completed_runs = max_completed_runs

    def publish(self, run_id: str, event: str, **payload: Any) -> tuple[int, dict[str, Any]]:
        event_payload = {"event": event, **payload}
        seq = self._next_seq[run_id]
        self._next_seq[run_id] += 1
        self._events[run_id].append(event_payload)
        self._seqs[run_id].append(seq)
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
        return seq, dict(event_payload)

    def snapshot(self, run_id: str) -> list[dict[str, Any]]:
        return [dict(event) for event in self._events.get(run_id, [])]

    def snapshot_records(self, run_id: str, *, after_seq: int = 0) -> list[tuple[int, dict[str, Any]]]:
        return [
            (seq, dict(event))
            for seq, event in zip(self._seqs.get(run_id, []), self._events.get(run_id, []), strict=False)
            if seq > after_seq
        ]

    def _prune_completed_runs(self) -> None:
        while len(self._completed_order) > self._max_completed_runs:
            stale_run_id = self._completed_order.popleft()
            self._completed_runs.discard(stale_run_id)
            self._events.pop(stale_run_id, None)
            self._seqs.pop(stale_run_id, None)
            self._signals.pop(stale_run_id, None)

    async def stream(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        async for _, event in self.stream_records(run_id):
            yield event

    async def stream_records(self, run_id: str, *, after_seq: int = 0) -> AsyncIterator[tuple[int, dict[str, Any]]]:
        seen = 0
        while True:
            events = self._events.get(run_id, [])
            seqs = self._seqs.get(run_id, [])
            while seen < len(seqs) and seqs[seen] <= after_seq:
                seen += 1
            while seen < len(events):
                seq = seqs[seen]
                event = dict(events[seen])
                seen += 1
                yield seq, event
            if run_id in self._completed_runs:
                return
            signal = self._signals[run_id]
            signal.clear()
            await signal.wait()


run_broker = RunEventBroker()
