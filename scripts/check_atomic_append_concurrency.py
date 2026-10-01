"""Cross-dialect concurrency check for the atomic stream append API.

Runs N concurrent appends against the same run and thread on the given
metadata backend and asserts every seq is unique and gapless, and that the
event table contains exactly N rows (no frame lost, no duplicate). This is the
sqlite-independent proof of the counter-row locking claim: MySQL/PostgreSQL
honour SELECT ... FOR UPDATE (sqlite does not), so the concurrent publishers
must serialize on the counter row rather than rely on uniqueness retries.

Usage (repo root, from WSL2):
  UV_PROJECT_ENVIRONMENT=... uv run python scripts/check_atomic_append_concurrency.py mysql|postgresql|sqlite
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select  # noqa: E402

from agentseek_api.core.database import db_manager  # noqa: E402
from agentseek_api.core.orm import RunStreamEvent, ThreadStreamEvent  # noqa: E402
from agentseek_api.services import stream_persistence as stream_module  # noqa: E402
from agentseek_api.settings import settings  # noqa: E402

_N = 12


async def _count_events(scope: str, scope_id: str) -> int:
    model = RunStreamEvent if scope == "run" else ThreadStreamEvent
    id_column = model.run_id if scope == "run" else model.thread_id
    async with db_manager.get_session_factory()() as session:
        return int(await session.scalar(select(func.count()).select_from(model).where(id_column == scope_id)))


async def _exercise(backend: str) -> None:
    # Unique scope ids per run so a pre-existing counter row (from an earlier
    # invocation against the same database) cannot shift the expected range.
    stamp = int(time.time() * 1000)
    run_id, thread_id = f"run-conc-{backend}-{stamp}", f"thread-conc-{backend}-{stamp}"
    for scope, scope_id in (("run", run_id), ("thread", thread_id)):
        if scope == "run":
            append = stream_module.append_run_stream_event_atomic
            results = await asyncio.gather(*(append(scope_id, {"event": "message", "data": i}) for i in range(_N)))
            rows = await stream_module.load_run_stream_events(scope_id)
            seqs = sorted(seq for seq, _ in results)
            persisted = [seq for seq, _ in rows]
        else:
            append = stream_module.append_thread_stream_event_atomic
            payload = lambda i: {"method": "values", "params": {"namespace": [], "timestamp": 1, "data": i}}  # noqa: E731
            results = await asyncio.gather(*(append(scope_id, payload(i)) for i in range(_N)))
            rows = await stream_module.load_thread_stream_events(scope_id, channels=["values"], namespaces=None, depth=None)
            seqs = sorted(seq for seq, _ in results)
            persisted = [event["seq"] for event in rows]

        assert seqs == list(range(1, _N + 1)), f"[{backend}/{scope}] non-gapless seqs: {seqs}"
        assert persisted == list(range(1, _N + 1)), f"[{backend}/{scope}] persisted mismatch: {persisted}"
        assert len(results) == _N, f"[{backend}/{scope}] frame dropped: got {len(results)}"
        count = await _count_events(scope, scope_id)
        assert count == _N, f"[{backend}/{scope}] event rows != {_N}: {count}"
        print(f"[{backend}/{scope}] OK: {_N} concurrent appends -> unique gapless 1..{_N}, {count} rows")

    print(f"[{backend}] PASS")


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "sqlite"
    if backend == "sqlite":
        settings.SEEKDB_EMBED = False
        settings.SEEKDB_URL = "sqlite+aiosqlite:////tmp/atomic-conc.db"
        settings.METADATA_DB_BACKEND = "sqlite"
    elif backend == "mysql":
        settings.SEEKDB_EMBED = False
        settings.SEEKDB_URL = "mysql+aiomysql://root:root@127.0.0.1:33306/agentseek"
        settings.METADATA_DB_BACKEND = "mysql"
    elif backend == "postgresql":
        settings.SEEKDB_EMBED = False
        settings.SEEKDB_URL = "postgresql://postgres:postgres@127.0.0.1:35432/agentseek"
        settings.METADATA_DB_BACKEND = "postgresql"
    else:
        raise SystemExit(f"unknown backend: {backend}")
    asyncio.run(_run(backend))


async def _run(backend: str) -> None:
    # The concurrency check only exercises the metadata-DB stream events; the
    # langgraph checkpointer/store backends (OceanBase/MySQL via pymysql)
    # connect eagerly in their constructors and would fail under a postgres
    # metadata URL, so they are replaced with inert stand-ins.
    import agentseek_api.core.database as database_module

    class _FakeCheckpointer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def setup(self) -> None:
            return None

        def save_checkpoint(self, **kwargs: object) -> None:
            return None

    class _FakeStore:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    database_module.OceanBaseCheckpointSaver = _FakeCheckpointer  # type: ignore[assignment]
    database_module.LangGraphOceanBaseCheckpointSaver = _FakeCheckpointer  # type: ignore[assignment]
    database_module.OceanBaseStore = _FakeStore  # type: ignore[assignment]
    await db_manager.initialize()
    try:
        await _exercise(backend)
    finally:
        await db_manager.close()
        print(f"[{backend}] db closed")


if __name__ == "__main__":
    main()
