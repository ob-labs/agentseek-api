from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from redis import Redis


PROBE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "verify_worker_concurrency.py"


@pytest.fixture
def probe_module() -> Iterator[ModuleType]:
    spec = importlib.util.spec_from_file_location("verify_worker_concurrency", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(spec.name, None)


class FakeProbeClient:
    def __init__(
        self,
        *,
        queue_snapshots: list[Any],
        statuses: dict[str, str | list[str]] | None = None,
    ) -> None:
        self.queue_snapshots = queue_snapshots
        self.statuses = statuses or {}
        self.created: list[tuple[str, float, bool]] = []

    def queue_snapshot(self) -> Any:
        if len(self.queue_snapshots) > 1:
            return self.queue_snapshots.pop(0)
        return self.queue_snapshots[0]

    def run_status(self, run: Any) -> str:
        value = self.statuses[run.run_id]
        if isinstance(value, list):
            if len(value) > 1:
                return value.pop(0)
            return value[0]
        return value

    def create_stress_run(self, name: str, *, delay_seconds: float, fail: bool = False) -> Any:
        self.created.append((name, delay_seconds, fail))
        return self.run_ref_type(thread_id=f"thread-{name}", run_id=name)


def _client(probe_module: ModuleType, **kwargs: Any) -> FakeProbeClient:
    client = FakeProbeClient(**kwargs)
    client.run_ref_type = probe_module.RunRef
    return client


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_wait_for_queue_shape_rejects_processing_above_limit(probe_module: ModuleType) -> None:
    clock = FakeClock()
    client = _client(
        probe_module,
        queue_snapshots=[probe_module.QueueSnapshot(pending=0, processing=3)],
    )

    with pytest.raises(AssertionError, match="exceeded concurrency limit 2"):
        probe_module.wait_for_queue_shape(
            client,
            concurrency=2,
            minimum_pending=1,
            timeout_seconds=0.01,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )


def test_wait_for_queue_shape_polls_until_expected_shape(probe_module: ModuleType) -> None:
    clock = FakeClock()
    client = _client(
        probe_module,
        queue_snapshots=[
            probe_module.QueueSnapshot(pending=2, processing=1),
            probe_module.QueueSnapshot(pending=1, processing=2),
        ],
    )

    snapshot = probe_module.wait_for_queue_shape(
        client,
        concurrency=2,
        minimum_pending=1,
        timeout_seconds=0.2,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert snapshot == probe_module.QueueSnapshot(pending=1, processing=2)


def test_wait_for_queue_shape_timeout_reports_last_snapshot(probe_module: ModuleType) -> None:
    clock = FakeClock()
    client = _client(
        probe_module,
        queue_snapshots=[probe_module.QueueSnapshot(pending=0, processing=1)],
    )

    with pytest.raises(
        AssertionError,
        match=r"processing=2, pending>=1.*QueueSnapshot\(pending=0, processing=1\)",
    ):
        probe_module.wait_for_queue_shape(
            client,
            concurrency=2,
            minimum_pending=1,
            timeout_seconds=0.1,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )


def test_wait_for_status_reports_last_observed_status_on_timeout(probe_module: ModuleType) -> None:
    clock = FakeClock()
    run = probe_module.RunRef("thread-1", "run-1")
    client = _client(
        probe_module,
        queue_snapshots=[probe_module.QueueSnapshot(0, 0)],
        statuses={"run-1": "running"},
    )

    with pytest.raises(AssertionError, match=r"run-1.*running.*success"):
        probe_module.wait_for_status(
            client,
            run,
            expected_status="success",
            concurrency=2,
            timeout_seconds=0.1,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )


def test_wait_for_status_rejects_processing_above_limit_before_success(
    probe_module: ModuleType,
) -> None:
    run = probe_module.RunRef("thread-1", "run-1")
    client = _client(
        probe_module,
        queue_snapshots=[probe_module.QueueSnapshot(pending=0, processing=3)],
        statuses={"run-1": "success"},
    )

    with pytest.raises(AssertionError, match="exceeded concurrency limit 2"):
        probe_module.wait_for_status(
            client,
            run,
            expected_status="success",
            concurrency=2,
            timeout_seconds=1.0,
            sleep=lambda _: None,
        )


def test_validate_recovery_statuses_requires_expected_terminal_states(probe_module: ModuleType) -> None:
    client = _client(
        probe_module,
        queue_snapshots=[probe_module.QueueSnapshot(0, 0)],
        statuses={"r1": "error", "r2": "success", "r3": "running"},
    )

    with pytest.raises(AssertionError, match="queued"):
        probe_module.validate_statuses(
            client,
            {
                "failed": probe_module.RunRef("t1", "r1"),
                "long": probe_module.RunRef("t2", "r2"),
                "queued": probe_module.RunRef("t3", "r3"),
            },
            {"failed": "error", "long": "success", "queued": "success"},
            concurrency=2,
        )


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


class FakeHttpClient:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, object]]] = []
        self.gets: list[str] = []

    def post(self, path: str, *, json: dict[str, object]) -> FakeResponse:
        self.posts.append((path, json))
        if path == "/assistants":
            return FakeResponse({"assistant_id": "assistant-1"})
        if path == "/threads":
            return FakeResponse({"thread_id": "thread-1"})
        return FakeResponse({"thread_id": "thread-1", "run_id": "run-1"})

    def get(self, path: str) -> FakeResponse:
        self.gets.append(path)
        return FakeResponse({"status": "running"})

    def close(self) -> None:
        return None


class FakeRedisClient:
    def close(self) -> None:
        return None


def test_probe_client_uses_stress_graph_http_contract(probe_module: ModuleType) -> None:
    http_client = FakeHttpClient()
    redis_client = FakeRedisClient()
    client = probe_module.ProbeClient(
        base_url="http://127.0.0.1:2024",
        redis_url="redis://127.0.0.1:6379/0",
        http_client=http_client,
        redis_client=redis_client,
    )

    run = client.create_stress_run("bounded-long", delay_seconds=3.5, fail=False)

    assert run == probe_module.RunRef("thread-1", "run-1")
    assert http_client.posts[0][0] == "/assistants"
    assert http_client.posts[0][1]["graph_id"] == "stress_test"
    assert http_client.posts[1] == (
        "/threads",
        {"metadata": {"suite": "worker-concurrency", "case": "bounded-long"}},
    )
    assert http_client.posts[2] == (
        "/threads/thread-1/runs",
        {
            "assistant_id": "assistant-1",
            "input": {"delay": 3.5, "steps": 1, "fail": False},
        },
    )
    assert client.run_status(run) == "running"
    assert http_client.gets == ["/threads/thread-1/runs/run-1"]


def test_probe_client_reads_both_queue_lengths_in_one_atomic_redis_command(
    probe_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[object, ...]] = []
    redis_client = Redis()

    def execute_command(*args: object, **_options: object) -> object:
        commands.append(args)
        if args[0] == "EVAL":
            return [4, 2]
        if args[0] == "LLEN":
            return 4 if args[1] == "agentseek:runs:pending" else 2
        raise AssertionError(f"unexpected Redis command: {args!r}")

    monkeypatch.setattr(redis_client, "execute_command", execute_command)
    client = probe_module.ProbeClient(
        base_url="http://127.0.0.1:2024",
        redis_url="redis://127.0.0.1:6379/0",
        http_client=FakeHttpClient(),
        redis_client=redis_client,
    )

    assert client.queue_snapshot() == probe_module.QueueSnapshot(pending=4, processing=2)
    assert len(commands) == 1
    command = commands[0]
    assert command[0] == "EVAL"
    assert command[2:] == (
        2,
        "agentseek:runs:pending",
        "agentseek:runs:processing",
    )
    script = command[1]
    assert isinstance(script, str)
    assert script.count("redis.call('LLEN'") == 2


def test_bounded_probe_proves_refill_before_long_run_finishes(probe_module: ModuleType) -> None:
    clock = FakeClock()
    client = _client(
        probe_module,
        queue_snapshots=[
            probe_module.QueueSnapshot(pending=1, processing=2),
            probe_module.QueueSnapshot(pending=0, processing=0),
        ],
        statuses={
            "long": ["running", "success", "success"],
            "refill": ["success", "success"],
            "queued": ["success", "success"],
        },
    )

    runs = probe_module.run_bounded_probe(
        client,
        timeout_seconds=0.01,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert list(runs) == ["long", "refill", "queued"]
    assert client.created == [
        ("long", 3.0, False),
        ("refill", 1.0, False),
        ("queued", 0.1, False),
    ]


def test_bounded_probe_rejects_post_shape_processing_over_cap(probe_module: ModuleType) -> None:
    client = _client(
        probe_module,
        queue_snapshots=[
            probe_module.QueueSnapshot(pending=1, processing=2),
            probe_module.QueueSnapshot(pending=0, processing=3),
            probe_module.QueueSnapshot(pending=0, processing=0),
        ],
        statuses={
            "long": ["running", "success", "success"],
            "refill": ["success", "success"],
            "queued": ["success", "success"],
        },
    )

    with pytest.raises(AssertionError, match="exceeded concurrency limit 2"):
        probe_module.run_bounded_probe(client, timeout_seconds=1.0, sleep=lambda _: None)


def test_fanout_probe_submits_twelve_runs_against_ten_slots(probe_module: ModuleType) -> None:
    clock = FakeClock()
    client = _client(
        probe_module,
        queue_snapshots=[
            probe_module.QueueSnapshot(pending=2, processing=10),
            probe_module.QueueSnapshot(pending=0, processing=0),
        ],
        statuses={f"fanout-{index}": "success" for index in range(12)},
    )

    runs = probe_module.run_fanout_probe(
        client,
        timeout_seconds=0.01,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert len(runs) == 12
    assert client.created == [
        *[(f"fanout-{index}", 6.0, False) for index in range(10)],
        ("fanout-10", 0.1, False),
        ("fanout-11", 0.1, False),
    ]


def test_failure_probe_requires_error_and_sibling_successes(probe_module: ModuleType) -> None:
    clock = FakeClock()
    client = _client(
        probe_module,
        queue_snapshots=[
            probe_module.QueueSnapshot(pending=1, processing=2),
            probe_module.QueueSnapshot(pending=0, processing=0),
        ],
        statuses={"failed": "error", "long": "success", "queued": "success"},
    )

    runs = probe_module.run_failure_probe(
        client,
        timeout_seconds=0.01,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert list(runs) == ["failed", "long", "queued"]
    assert client.created == [
        ("failed", 1.5, True),
        ("long", 3.0, False),
        ("queued", 0.1, False),
    ]


def test_shutdown_seed_and_check_cover_two_inflight_and_one_queued(probe_module: ModuleType) -> None:
    clock = FakeClock()
    seed_client = _client(
        probe_module,
        queue_snapshots=[probe_module.QueueSnapshot(pending=1, processing=2)],
        statuses={"long-a": "running", "long-b": "running"},
    )
    runs = probe_module.seed_shutdown_probe(
        seed_client,
        timeout_seconds=0.01,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert list(runs) == ["long-a", "long-b", "queued"]
    assert seed_client.created == [
        ("long-a", 6.0, False),
        ("long-b", 6.0, False),
        ("queued", 0.1, False),
    ]

    check_client = _client(
        probe_module,
        queue_snapshots=[probe_module.QueueSnapshot(pending=0, processing=0)],
        statuses={"long-a": "success", "long-b": "success", "queued": "success"},
    )
    probe_module.check_shutdown_probe(
        check_client,
        runs,
        timeout_seconds=0.01,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )


def test_shutdown_seed_requires_both_long_runs_to_still_be_running(
    probe_module: ModuleType,
) -> None:
    client = _client(
        probe_module,
        queue_snapshots=[probe_module.QueueSnapshot(pending=1, processing=2)],
        statuses={"long-a": "running", "long-b": "success"},
    )

    with pytest.raises(AssertionError, match=r"long-b='success'.*'running'"):
        probe_module.seed_shutdown_probe(client, timeout_seconds=1.0, sleep=lambda _: None)


def test_wait_for_queues_empty_timeout_reports_last_snapshot(probe_module: ModuleType) -> None:
    clock = FakeClock()
    client = _client(
        probe_module,
        queue_snapshots=[probe_module.QueueSnapshot(pending=2, processing=1)],
    )

    with pytest.raises(
        AssertionError,
        match=r"did not drain.*QueueSnapshot\(pending=2, processing=1\)",
    ):
        probe_module.wait_for_queues_empty(
            client,
            concurrency=2,
            timeout_seconds=0.1,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )


def test_wait_for_queues_empty_rejects_processing_above_limit(probe_module: ModuleType) -> None:
    client = _client(
        probe_module,
        queue_snapshots=[probe_module.QueueSnapshot(pending=0, processing=3)],
    )

    with pytest.raises(AssertionError, match="exceeded concurrency limit 2"):
        probe_module.wait_for_queues_empty(
            client,
            concurrency=2,
            timeout_seconds=1.0,
            sleep=lambda _: None,
        )


def test_load_runs_reads_serialized_restart_state(
    probe_module: ModuleType,
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "restart.json"
    state_file.write_text(
        '{"mode":"shutdown-seed","runs":{"long-a":{"thread_id":"t-a","run_id":"r-a"}}}',
        encoding="utf-8",
    )

    assert probe_module._load_runs(state_file) == {
        "long-a": probe_module.RunRef(thread_id="t-a", run_id="r-a")
    }


def test_load_runs_rejects_non_object_run_state(
    probe_module: ModuleType,
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "invalid.json"
    state_file.write_text('{"runs":[]}', encoding="utf-8")

    with pytest.raises(SystemExit, match="runs object"):
        probe_module._load_runs(state_file)


def test_main_emits_one_json_document_and_no_other_output(
    probe_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = {"long": probe_module.RunRef(thread_id="thread-1", run_id="run-1")}

    class MainProbeClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> MainProbeClient:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    monkeypatch.setattr(probe_module, "ProbeClient", MainProbeClient)
    monkeypatch.setattr(probe_module, "run_bounded_probe", lambda *_args, **_kwargs: runs)

    result = probe_module.main(
        [
            "--base-url",
            "http://127.0.0.1:2024",
            "--redis-url",
            "redis://127.0.0.1:6379/0",
            "--mode",
            "bounded",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert captured.out == (
        '{"mode": "bounded", "runs": {"long": {"run_id": "run-1", '
        '"thread_id": "thread-1"}}}\n'
    )


@pytest.mark.parametrize("mode", ["bounded", "fanout", "failure", "shutdown-seed"])
def test_cli_accepts_probe_modes_without_state_file(probe_module: ModuleType, mode: str) -> None:
    args = probe_module.parse_args(
        [
            "--base-url",
            "http://127.0.0.1:2024",
            "--redis-url",
            "redis://127.0.0.1:6379/0",
            "--mode",
            mode,
        ]
    )

    assert args.mode == mode


def test_shutdown_check_cli_requires_state_file(probe_module: ModuleType) -> None:
    with pytest.raises(SystemExit) as exc_info:
        probe_module.parse_args(
            [
                "--base-url",
                "http://127.0.0.1:2024",
                "--redis-url",
                "redis://127.0.0.1:6379/0",
                "--mode",
                "shutdown-check",
            ]
        )

    assert exc_info.value.code == 2
