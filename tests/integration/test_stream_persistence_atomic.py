"""Regression tests for the atomic stream append API.

Covers the fourth-round review findings 1 and 2 (non-Redis sequence allocation
is not atomic; a live cursor can be exposed before it is durable):

- concurrent publishers of the same run/thread must allocate unique, gapless
  seqs (previously ``SELECT MAX(seq)+1`` races produced ``[1, 1]`` and one
  frame was silently dropped by the persistence helper)
- a failed durable append must raise (not be swallowed) so the event is never
  exposed to a client
- the per-stream counter row self-heals from ``MAX(seq)`` if it is deleted out
  from under an active stream (the two-phase delete path removes business rows
  before stream rows)

The tests run against the real sqlite metadata DB initialized by the ``client``
fixture (the same backend used by the regular integration suite), which is the
weakest backend: if atomic allocation is correct on sqlite's single-writer
locking, it is correct on the row-locked MySQL/SeekDB and PostgreSQL backends.
"""

from __future__ import annotations

import asyncio

import pytest

from agentseek_api.services import stream_persistence as stream_module


def _thread_payload(**params: object) -> dict[str, object]:
    return {
        "method": "values",
        "params": {
            "namespace": [],
            "timestamp": 1,
            **params,
        },
    }


def test_concurrent_atomic_appends_allocate_unique_seqs(client) -> None:
    """N concurrent publishers of the same run/thread each get a unique seq.

    Regression for the review reproduction ``[1, 1]``: the old
    ``SELECT MAX(seq)+1`` allocator let two publishers observe the same max and
    the unique constraint then dropped one frame. The counter-row append must
    serialize publishers and hand out exactly ``1..N``.
    """

    async def run_concurrent(scope: str, scope_id: str, n: int) -> list[int]:
        if scope == "run":
            append = stream_module.append_run_stream_event_atomic
            load = stream_module.load_run_stream_events
            results = await asyncio.gather(
                *(append(scope_id, {"event": "message", "data": i}) for i in range(n))
            )
            rows = await load(scope_id)
            return sorted(seq for seq, _ in results), [seq for seq, _ in rows]

        append = stream_module.append_thread_stream_event_atomic
        load = lambda: stream_module.load_thread_stream_events(  # noqa: E731
            scope_id, channels=["values"], namespaces=None, depth=None
        )
        results = await asyncio.gather(
            *(append(scope_id, _thread_payload(data=i)) for i in range(n))
        )
        rows = await load()
        return sorted(seq for seq, _ in results), [event["seq"] for event in rows]

    run_seqs, run_rows = client.portal.call(run_concurrent, "run", "run-concurrent", 8)
    assert run_seqs == list(range(1, 9)), f"run seqs must be unique and gapless: {run_seqs}"
    assert run_rows == list(range(1, 9)), f"persisted run rows must match: {run_rows}"

    thread_seqs, thread_rows = client.portal.call(run_concurrent, "thread", "thread-concurrent", 8)
    assert thread_seqs == list(range(1, 9)), f"thread seqs must be unique and gapless: {thread_seqs}"
    assert thread_rows == list(range(1, 9)), f"persisted thread rows must match: {thread_rows}"


def test_atomic_append_raises_on_db_failure_and_is_not_exposed(client, monkeypatch) -> None:
    """A failed durable append must raise instead of being swallowed.

    The caller (``_publish_run_event`` and friends) only publishes to the
    in-memory broker after the append succeeds, so a client can never receive a
    seq that was not durably committed.
    """
    from agentseek_api.services import run_jobs as run_jobs_module
    from agentseek_api.services.run_state import run_broker

    async def failing_stage(*_args: object, **_kwargs: object):
        raise RuntimeError("durable append failed")

    async def exercise() -> list[str]:
        published: list[str] = []
        original_publish = run_broker.publish

        def tracking_publish(run_id: str, event: str, **payload: object):
            published.append(f"{run_id}:{event}")
            return original_publish(run_id, event, **payload)

        monkeypatch.setattr(run_broker, "publish", tracking_publish)
        monkeypatch.setattr(
            stream_module, "_stage_db_event", failing_stage, raising=False
        )
        try:
            await run_jobs_module._publish_run_event("run-fail-inject", "start")
        except RuntimeError:
            return ["raised"] + published
        return ["no-raise"] + published

    outcome = client.portal.call(exercise)
    assert outcome[0] == "raised", "append failure must propagate to the caller"
    assert len(outcome) == 1, "the event must not be exposed to the broker on failure"


def test_atomic_append_self_heals_after_counter_row_deleted(client) -> None:
    """The counter row re-seeds from ``MAX(seq)`` if it disappears mid-stream.

    The two-phase delete path (``threads.py``) removes business rows before
    stream rows, so a counter row can be deleted while a stream is still being
    appended to. The append must not collide with persisted rows - it re-seeds
    from the durable events and keeps allocating monotonically.
    """
    from sqlalchemy import delete

    from agentseek_api.core.database import db_manager
    from agentseek_api.core.orm import StreamSequence

    async def exercise() -> list[tuple[int, str]]:
        first = await stream_module.append_run_stream_event_atomic(
            "run-self-heal", {"event": "start"}
        )
        second = await stream_module.append_run_stream_event_atomic(
            "run-self-heal", {"event": "message", "data": "mid"}
        )
        async with db_manager.get_session_factory()() as session:
            await session.execute(
                delete(StreamSequence).where(
                    StreamSequence.scope == "run", StreamSequence.scope_id == "run-self-heal"
                )
            )
            await session.commit()
        third = await stream_module.append_run_stream_event_atomic(
            "run-self-heal", {"event": "end", "status": "success"}
        )
        rows = await stream_module.load_run_stream_events("run-self-heal")
        return (
            [first[0], second[0], third[0]],
            [(seq, str(payload.get("event"))) for seq, payload in rows],
        )

    seqs, rows = client.portal.call(exercise)
    assert seqs == [1, 2, 3], f"seq must stay monotonic across counter deletion: {seqs}"
    assert rows == [(1, "start"), (2, "message"), (3, "end")], (
        f"persisted events must be gapless and ordered: {rows}"
    )


def test_atomic_append_rejects_explicit_cursor_collision(client) -> None:
    """A requested cursor must not be silently renumbered or overwrite history."""
    from sqlalchemy.exc import IntegrityError

    async def exercise():
        await stream_module.append_run_stream_event_atomic("collision", {"event": "start"})
        with pytest.raises(IntegrityError):
            await stream_module.append_run_stream_event_atomic("collision", {"event": "end"}, seq=1)
        return await stream_module.load_run_stream_events("collision")

    assert client.portal.call(exercise) == [(1, {"event": "start"})]
