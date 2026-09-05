import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from agentseek_api.services import stream_persistence as stream_module


class FakeScalarResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows


class FakeSession:
    def __init__(
        self,
        *,
        scalar_values: list[object | None] | None = None,
        scalars_rows: list[object] | None = None,
        execute_error: Exception | None = None,
        commit_error: Exception | None = None,
    ) -> None:
        self.scalar_values = list(scalar_values or [])
        self.scalars_rows = list(scalars_rows or [])
        self.execute_error = execute_error
        self.commit_error = commit_error
        self.added: list[object] = []
        self.commits = 0

    async def scalar(self, _query: Any) -> object | None:
        return self.scalar_values.pop(0) if self.scalar_values else None

    async def scalars(self, _query: Any) -> FakeScalarResult:
        return FakeScalarResult(self.scalars_rows)

    async def execute(self, _query: Any) -> None:
        if self.execute_error is not None:
            raise self.execute_error

    async def commit(self) -> None:
        self.commits += 1
        if self.commit_error is not None:
            raise self.commit_error

    def add(self, obj: object) -> None:
        self.added.append(obj)


class FakeSessionContext:
    def __init__(self, session: FakeSession) -> None:
        self.session = session

    async def __aenter__(self) -> FakeSession:
        return self.session

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        return None


class FakeSessionFactory:
    def __init__(self, sessions: list[FakeSession]) -> None:
        self.sessions = sessions

    def __call__(self) -> FakeSessionContext:
        return FakeSessionContext(self.sessions.pop(0))


class FakeRedisStream:
    def __init__(self) -> None:
        self.entries: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.counts: dict[str, int] = {}
        self.eval_calls: list[tuple[str, str]] = []
        self.xadd_calls: list[tuple[str, str, int, bool]] = []
        self.expirations: dict[str, int] = {}
        self.deleted_keys: list[str] = []

    async def xadd(
        self,
        key: str,
        fields: dict[str, str],
        *,
        id: str,
        maxlen: int,
        approximate: bool,
    ) -> str:
        self.entries.setdefault(key, []).append((id, dict(fields)))
        self.xadd_calls.append((key, id, maxlen, approximate))
        return id

    async def xrange(self, key: str, *, min: str, max: str) -> list[tuple[str, dict[str, str]]]:
        assert max == "+"
        after_seq = int(min.removeprefix("(").split("-", 1)[0])
        return [
            (entry_id, fields)
            for entry_id, fields in self.entries.get(key, [])
            if int(entry_id.split("-", 1)[0]) > after_seq
        ]

    async def expire(self, key: str, seconds: int) -> bool:
        self.expirations[key] = seconds
        return True

    async def delete(self, *keys: str) -> int:
        self.deleted_keys.extend(keys)
        for key in keys:
            self.entries.pop(key, None)
        return len(keys)

    async def eval(self, script: str, numkeys: int, *args: str) -> list[object]:
        assert numkeys == 2
        assert "XADD" in script
        seq_key, stream_key, encoded_payload, maxlen, ttl_seconds, event_prefix = args
        seq = self.counts.get(seq_key, 0) + 1
        self.counts[seq_key] = seq
        payload = json.loads(encoded_payload)
        if event_prefix:
            payload = {"type": "event", "event_id": f"{event_prefix}:{seq}", "seq": seq, **payload}
        encoded_event = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.entries.setdefault(stream_key, []).append((f"{seq}-0", {"payload": encoded_event}))
        self.expirations[stream_key] = int(ttl_seconds)
        self.eval_calls.append((seq_key, stream_key))
        _ = maxlen
        return [seq, encoded_event]


@pytest.mark.asyncio
async def test_redis_atomic_run_stream_events_bypass_metadata_db(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = FakeRedisStream()
    monkeypatch.setattr(stream_module.settings, "EXECUTOR_BACKEND", "redis")
    monkeypatch.setattr(stream_module, "_redis_client", redis)
    monkeypatch.setattr(
        stream_module,
        "_metadata_db_ready",
        lambda: (_ for _ in ()).throw(AssertionError("metadata DB must not be used")),
    )

    seq, _ = await stream_module.append_redis_run_stream_event("run-1", {"event": "end"})
    events = await stream_module.load_run_stream_events("run-1")

    assert seq == 1
    assert events == [(1, {"event": "end"})]
    assert redis.eval_calls == [("agentseek:runs:stream-seq:run-1", "agentseek:runs:stream:run-1")]
    assert redis.expirations["agentseek:runs:stream:run-1"] == 3600


@pytest.mark.asyncio
async def test_redis_atomic_append_allocates_sequence_and_persists_event(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = FakeRedisStream()
    monkeypatch.setattr(stream_module.settings, "EXECUTOR_BACKEND", "redis")
    monkeypatch.setattr(stream_module, "_redis_client", redis)
    append_run = getattr(stream_module, "append_redis_run_stream_event", None)
    append_thread = getattr(stream_module, "append_redis_thread_stream_event", None)

    assert callable(append_run)
    assert callable(append_thread)
    run_seq, run_event = await append_run("run-1", {"event": "message", "data": "one"})
    thread_seq, thread_event = await append_thread(
        "thread-1",
        {"method": "values", "params": {"namespace": [], "data": {"ok": True}}},
    )

    assert (run_seq, run_event) == (1, {"event": "message", "data": "one"})
    assert thread_seq == 1
    assert thread_event["seq"] == 1
    assert thread_event["event_id"] == "thread-1:1"
    assert redis.eval_calls == [
        ("agentseek:runs:stream-seq:run-1", "agentseek:runs:stream:run-1"),
        ("agentseek:threads:stream-seq:thread-1", "agentseek:threads:stream:thread-1"),
    ]


@pytest.mark.asyncio
async def test_redis_load_failure_is_not_treated_as_empty_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingRedis(FakeRedisStream):
        async def xrange(self, key: str, *, min: str, max: str) -> list[tuple[str, dict[str, str]]]:
            _ = (key, min, max)
            raise RuntimeError("redis unavailable")

    monkeypatch.setattr(stream_module.settings, "EXECUTOR_BACKEND", "redis")
    monkeypatch.setattr(stream_module, "_redis_client", FailingRedis())

    with pytest.raises(RuntimeError, match="redis unavailable"):
        await stream_module.load_run_stream_events("run-1")


@pytest.mark.asyncio
async def test_redis_delete_failure_emits_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    class FailingRedis(FakeRedisStream):
        async def delete(self, *keys: str) -> int:
            _ = keys
            raise RuntimeError("redis unavailable")

    monkeypatch.setattr(stream_module.settings, "EXECUTOR_BACKEND", "redis")
    monkeypatch.setattr(stream_module, "_redis_client", FailingRedis())

    with caplog.at_level(logging.WARNING):
        await stream_module.delete_run_stream_events(["run-1"])

    assert "delete Redis run stream keys" in caplog.text


@pytest.mark.asyncio
async def test_redis_thread_stream_events_filter_and_bypass_metadata_db(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = FakeRedisStream()
    monkeypatch.setattr(stream_module.settings, "EXECUTOR_BACKEND", "redis")
    monkeypatch.setattr(stream_module, "_redis_client", redis)
    monkeypatch.setattr(
        stream_module,
        "_metadata_db_ready",
        lambda: (_ for _ in ()).throw(AssertionError("metadata DB must not be used")),
    )
    values_payload = {
        "method": "values",
        "params": {"namespace": ["child"], "data": {"ok": True}},
    }
    updates_payload = {
        "method": "updates",
        "params": {"namespace": ["child"], "data": {"ignored": True}},
    }

    await stream_module.append_redis_thread_stream_event("thread-1", values_payload)
    await stream_module.append_redis_thread_stream_event("thread-1", updates_payload)
    events = await stream_module.load_thread_stream_events(
        "thread-1",
        channels=["values"],
        namespaces=[["child"]],
        depth=None,
        after_seq=0,
    )

    assert len(events) == 1
    assert events[0]["seq"] == 1
    assert events[0]["method"] == "values"
    assert events[0]["params"] == values_payload["params"]


@pytest.mark.asyncio
async def test_redis_legacy_persist_helpers_do_not_append_non_atomic_events(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    redis = FakeRedisStream()
    session = FakeSession()
    monkeypatch.setattr(stream_module.settings, "EXECUTOR_BACKEND", "redis")
    monkeypatch.setattr(stream_module, "_redis_client", redis)

    with caplog.at_level(logging.WARNING):
        await stream_module.persist_run_stream_event("run-1", seq=3, payload={"event": "message"})
        await stream_module.add_run_stream_event_to_session(
            session,
            "run-1",
            seq=4,
            payload={"event": "end"},
        )
        await stream_module.add_thread_stream_event_to_session(
            session,
            "thread-1",
            seq=5,
            payload={"seq": 5, "method": "lifecycle"},
        )
        await stream_module.persist_thread_stream_event(
            "thread-1",
            {"seq": 6, "method": "values"},
        )

    assert session.added == []
    assert redis.entries == {}
    assert caplog.text.count("non-atomic Redis stream append") == 4


@pytest.mark.asyncio
async def test_redis_loaders_skip_malformed_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = FakeRedisStream()
    redis.entries["agentseek:runs:stream:run-1"] = [
        ("1-0", {"payload": "not-json"}),
        ("2-0", {"payload": '{"event":"end"}'}),
    ]
    monkeypatch.setattr(stream_module.settings, "EXECUTOR_BACKEND", "redis")
    monkeypatch.setattr(stream_module, "_redis_client", redis)

    events = await stream_module.load_run_stream_events("run-1")

    assert events == [(2, {"event": "end"})]


@pytest.mark.asyncio
async def test_redis_delete_helpers_remove_stream_and_sequence_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = FakeRedisStream()
    monkeypatch.setattr(stream_module.settings, "EXECUTOR_BACKEND", "redis")
    monkeypatch.setattr(stream_module, "_redis_client", redis)
    monkeypatch.setattr(
        stream_module,
        "_metadata_db_ready",
        lambda: (_ for _ in ()).throw(AssertionError("metadata DB must not be used")),
    )

    await stream_module.delete_run_stream_events(["run-1", "run-2"])
    await stream_module.delete_thread_stream_events("thread-1")

    assert redis.deleted_keys == [
        "agentseek:runs:stream:run-1",
        "agentseek:runs:stream-seq:run-1",
        "agentseek:runs:stream:run-2",
        "agentseek:runs:stream-seq:run-2",
        "agentseek:threads:stream:thread-1",
        "agentseek:threads:stream-seq:thread-1",
    ]


def test_parse_last_event_id_handles_empty_and_invalid_values() -> None:
    assert stream_module.parse_last_event_id(None) is None
    assert stream_module.parse_last_event_id("") is None
    assert stream_module.parse_last_event_id("7") == 7
    assert stream_module.parse_last_event_id("-1") is None
    assert stream_module.parse_last_event_id("not-an-int") is None


@pytest.mark.asyncio
async def test_add_stream_event_helpers_skip_existing_records() -> None:
    session = FakeSession(scalar_values=[1, 1])

    await stream_module.add_run_stream_event_to_session(
        session,
        "run-1",
        seq=1,
        payload={"event": "start"},
    )
    await stream_module.add_thread_stream_event_to_session(
        session,
        "thread-1",
        seq=1,
        payload={"method": "lifecycle"},
    )

    assert session.added == []


@pytest.mark.asyncio
async def test_load_helpers_return_empty_when_metadata_db_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stream_module, "_metadata_db_ready", lambda: False)

    assert await stream_module.load_run_stream_events("run-1") == []
    assert await stream_module.load_thread_stream_events(
        "thread-1",
        channels=["values"],
        namespaces=None,
        depth=None,
    ) == []

    await stream_module.delete_run_stream_events([])


@pytest.mark.asyncio
async def test_load_helpers_return_empty_when_session_factory_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stream_module, "_metadata_db_ready", lambda: True)

    def raise_runtime_error() -> object:
        raise RuntimeError("not initialized")

    monkeypatch.setattr(stream_module.db_manager, "get_session_factory", raise_runtime_error)

    assert await stream_module.load_run_stream_events("run-1") == []
    assert await stream_module.load_thread_stream_events(
        "thread-1",
        channels=["values"],
        namespaces=None,
        depth=None,
    ) == []


@pytest.mark.asyncio
async def test_persist_helpers_swallow_commit_errors_and_invalid_thread_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_session = FakeSession(scalar_values=[None], commit_error=RuntimeError("boom"))
    thread_session = FakeSession(scalar_values=[None], commit_error=RuntimeError("boom"))
    session_factory = FakeSessionFactory([run_session, thread_session])

    monkeypatch.setattr(stream_module, "_metadata_db_ready", lambda: True)
    monkeypatch.setattr(stream_module.db_manager, "get_session_factory", lambda: session_factory)

    await stream_module.persist_run_stream_event("run-1", seq=1, payload={"event": "start"})
    await stream_module.persist_thread_stream_event("thread-1", {"seq": 0, "method": "lifecycle"})
    await stream_module.persist_thread_stream_event("thread-1", {"seq": 2, "method": "lifecycle"})

    assert len(run_session.added) == 1
    assert len(thread_session.added) == 1


@pytest.mark.asyncio
async def test_delete_helpers_swallow_session_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    session_factory = FakeSessionFactory(
        [
            FakeSession(execute_error=RuntimeError("boom")),
            FakeSession(execute_error=RuntimeError("boom")),
        ]
    )

    monkeypatch.setattr(stream_module, "_metadata_db_ready", lambda: True)
    monkeypatch.setattr(stream_module.db_manager, "get_session_factory", lambda: session_factory)

    await stream_module.delete_run_stream_events(["run-1"])
    await stream_module.delete_thread_stream_events("thread-1")


@pytest.mark.asyncio
async def test_load_thread_stream_events_normalizes_non_list_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = SimpleNamespace(
        payload_json={
            "seq": 1,
            "method": "values",
            "params": {"namespace": "not-a-list", "timestamp": 1, "data": {"ok": True}},
        }
    )
    session_factory = FakeSessionFactory([FakeSession(scalars_rows=[row])])

    monkeypatch.setattr(stream_module, "_metadata_db_ready", lambda: True)
    monkeypatch.setattr(stream_module.db_manager, "get_session_factory", lambda: session_factory)

    events = await stream_module.load_thread_stream_events(
        "thread-1",
        channels=["values"],
        namespaces=None,
        depth=None,
    )

    assert events == [row.payload_json]
