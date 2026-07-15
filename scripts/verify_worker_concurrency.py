from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from redis import Redis


PENDING_QUEUE_KEY = "agentseek:runs:pending"
PROCESSING_QUEUE_KEY = "agentseek:runs:processing"
TERMINAL_STATUSES = frozenset({"error", "interrupted", "success"})
POLL_INTERVAL_SECONDS = 0.05
DEFAULT_TIMEOUT_SECONDS = 20.0
QUEUE_SNAPSHOT_SCRIPT = """
return {
    redis.call('LLEN', KEYS[1]),
    redis.call('LLEN', KEYS[2])
}
""".strip()


@dataclass(frozen=True, slots=True)
class RunRef:
    thread_id: str
    run_id: str


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    pending: int
    processing: int


class ProbeOperations(Protocol):
    def create_stress_run(self, name: str, *, delay_seconds: float, fail: bool = False) -> RunRef: ...

    def run_status(self, run: RunRef) -> str: ...

    def queue_snapshot(self) -> QueueSnapshot: ...


class ProbeClient:
    def __init__(
        self,
        *,
        base_url: str,
        redis_url: str,
        http_client: Any | None = None,
        redis_client: Any | None = None,
    ) -> None:
        self._http = http_client or httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"x-user-id": "worker-concurrency-probe"},
            timeout=10.0,
            trust_env=False,
        )
        self._redis = redis_client or Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=5.0,
        )
        self._assistant_id: str | None = None

    def __enter__(self) -> ProbeClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()
        self._redis.close()

    @staticmethod
    def _response_object(response: Any, *, operation: str) -> dict[str, object]:
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise AssertionError(f"{operation} returned a non-object JSON response: {payload!r}")
        return payload

    def _stress_assistant_id(self) -> str:
        if self._assistant_id is None:
            response = self._http.post(
                "/assistants",
                json={"name": "worker-concurrency-probe", "graph_id": "stress_test"},
            )
            payload = self._response_object(response, operation="create stress_test assistant")
            assistant_id = payload.get("assistant_id")
            if not isinstance(assistant_id, str) or not assistant_id:
                raise AssertionError(f"assistant response omitted assistant_id: {payload!r}")
            self._assistant_id = assistant_id
        return self._assistant_id

    def create_stress_run(self, name: str, *, delay_seconds: float, fail: bool = False) -> RunRef:
        assistant_id = self._stress_assistant_id()
        thread_response = self._http.post(
            "/threads",
            json={"metadata": {"suite": "worker-concurrency", "case": name}},
        )
        thread_payload = self._response_object(thread_response, operation=f"create thread for {name}")
        thread_id = thread_payload.get("thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            raise AssertionError(f"thread response omitted thread_id: {thread_payload!r}")

        run_response = self._http.post(
            f"/threads/{thread_id}/runs",
            json={
                "assistant_id": assistant_id,
                "input": {"delay": delay_seconds, "steps": 1, "fail": fail},
            },
        )
        run_payload = self._response_object(run_response, operation=f"create stress run {name}")
        run_id = run_payload.get("run_id")
        response_thread_id = run_payload.get("thread_id")
        if not isinstance(run_id, str) or not run_id:
            raise AssertionError(f"run response omitted run_id: {run_payload!r}")
        if response_thread_id is not None and response_thread_id != thread_id:
            raise AssertionError(
                f"run response thread_id {response_thread_id!r} did not match created thread {thread_id!r}"
            )
        return RunRef(thread_id=thread_id, run_id=run_id)

    def run_status(self, run: RunRef) -> str:
        response = self._http.get(f"/threads/{run.thread_id}/runs/{run.run_id}")
        payload = self._response_object(response, operation=f"get run {run.run_id}")
        status = payload.get("status")
        if not isinstance(status, str) or not status:
            raise AssertionError(f"run response omitted status: {payload!r}")
        return status

    def queue_snapshot(self) -> QueueSnapshot:
        lengths = self._redis.eval(
            QUEUE_SNAPSHOT_SCRIPT,
            2,
            PENDING_QUEUE_KEY,
            PROCESSING_QUEUE_KEY,
        )
        if not isinstance(lengths, (list, tuple)) or len(lengths) != 2:
            raise AssertionError(f"atomic Redis queue snapshot returned invalid lengths: {lengths!r}")
        return QueueSnapshot(pending=int(lengths[0]), processing=int(lengths[1]))


def _queue_snapshot_within_cap(
    client: ProbeOperations,
    *,
    concurrency: int,
) -> QueueSnapshot:
    snapshot = client.queue_snapshot()
    if snapshot.processing > concurrency:
        raise AssertionError(
            f"processing count {snapshot.processing} exceeded concurrency limit {concurrency}"
        )
    return snapshot


def wait_for_queue_shape(
    client: ProbeOperations,
    *,
    concurrency: int,
    minimum_pending: int,
    timeout_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> QueueSnapshot:
    deadline = monotonic() + timeout_seconds
    last_snapshot: QueueSnapshot | None = None
    while monotonic() < deadline:
        last_snapshot = _queue_snapshot_within_cap(client, concurrency=concurrency)
        if last_snapshot.processing == concurrency and last_snapshot.pending >= minimum_pending:
            return last_snapshot
        sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(
        "queue did not reach the expected bounded-concurrency shape "
        f"(processing={concurrency}, pending>={minimum_pending}); last snapshot was {last_snapshot!r}"
    )


def wait_for_status(
    client: ProbeOperations,
    run: RunRef,
    *,
    expected_status: str,
    concurrency: int,
    timeout_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> str:
    deadline = monotonic() + timeout_seconds
    last_status: str | None = None
    while monotonic() < deadline:
        _queue_snapshot_within_cap(client, concurrency=concurrency)
        last_status = client.run_status(run)
        if last_status == expected_status:
            return last_status
        if last_status in TERMINAL_STATUSES:
            raise AssertionError(
                f"run {run.run_id} reached terminal status {last_status!r}; expected {expected_status!r}"
            )
        sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(
        f"run {run.run_id} timed out with last status {last_status!r}; expected {expected_status!r}"
    )


def validate_statuses(
    client: ProbeOperations,
    runs: Mapping[str, RunRef],
    expected_statuses: Mapping[str, str],
    *,
    concurrency: int,
) -> None:
    if runs.keys() != expected_statuses.keys():
        raise AssertionError(
            f"run labels {sorted(runs)} did not match expected labels {sorted(expected_statuses)}"
        )
    mismatches: list[str] = []
    for name, run in runs.items():
        _queue_snapshot_within_cap(client, concurrency=concurrency)
        actual = client.run_status(run)
        expected = expected_statuses[name]
        if actual != expected:
            mismatches.append(f"{name}={actual!r} (expected {expected!r})")
    if mismatches:
        raise AssertionError("unexpected run statuses: " + ", ".join(mismatches))


def wait_for_queues_empty(
    client: ProbeOperations,
    *,
    concurrency: int,
    timeout_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> QueueSnapshot:
    deadline = monotonic() + timeout_seconds
    last_snapshot: QueueSnapshot | None = None
    while monotonic() < deadline:
        last_snapshot = _queue_snapshot_within_cap(client, concurrency=concurrency)
        if last_snapshot == QueueSnapshot(pending=0, processing=0):
            return last_snapshot
        sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(f"Redis run queues did not drain; last snapshot was {last_snapshot!r}")


def _wait_for_expected_statuses(
    client: ProbeOperations,
    runs: Mapping[str, RunRef],
    expected_statuses: Mapping[str, str],
    *,
    concurrency: int,
    timeout_seconds: float,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> None:
    for name, run in runs.items():
        wait_for_status(
            client,
            run,
            expected_status=expected_statuses[name],
            concurrency=concurrency,
            timeout_seconds=timeout_seconds,
            sleep=sleep,
            monotonic=monotonic,
        )
    validate_statuses(client, runs, expected_statuses, concurrency=concurrency)


def run_bounded_probe(
    client: ProbeOperations,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, RunRef]:
    runs = {
        "long": client.create_stress_run("long", delay_seconds=3.0),
        "refill": client.create_stress_run("refill", delay_seconds=1.0),
        "queued": client.create_stress_run("queued", delay_seconds=0.1),
    }
    wait_for_queue_shape(
        client,
        concurrency=2,
        minimum_pending=1,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
        monotonic=monotonic,
    )
    wait_for_status(
        client,
        runs["queued"],
        expected_status="success",
        concurrency=2,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
        monotonic=monotonic,
    )
    _queue_snapshot_within_cap(client, concurrency=2)
    long_status = client.run_status(runs["long"])
    if long_status != "running":
        raise AssertionError(
            "queued run did not prove worker refill while the long run was active: "
            f"long run status was {long_status!r}"
        )
    expected = {name: "success" for name in runs}
    for name in ("refill", "long"):
        wait_for_status(
            client,
            runs[name],
            expected_status="success",
            concurrency=2,
            timeout_seconds=timeout_seconds,
            sleep=sleep,
            monotonic=monotonic,
        )
    validate_statuses(client, runs, expected, concurrency=2)
    wait_for_queues_empty(
        client,
        concurrency=2,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
        monotonic=monotonic,
    )
    return runs


def run_fanout_probe(
    client: ProbeOperations,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, RunRef]:
    runs: dict[str, RunRef] = {}
    for index in range(12):
        delay_seconds = 6.0 if index < 10 else 0.1
        name = f"fanout-{index}"
        runs[name] = client.create_stress_run(name, delay_seconds=delay_seconds)
    wait_for_queue_shape(
        client,
        concurrency=10,
        minimum_pending=2,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
        monotonic=monotonic,
    )
    expected = {name: "success" for name in runs}
    _wait_for_expected_statuses(
        client,
        runs,
        expected,
        concurrency=10,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
        monotonic=monotonic,
    )
    wait_for_queues_empty(
        client,
        concurrency=10,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
        monotonic=monotonic,
    )
    return runs


def run_failure_probe(
    client: ProbeOperations,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, RunRef]:
    runs = {
        "failed": client.create_stress_run("failed", delay_seconds=1.5, fail=True),
        "long": client.create_stress_run("long", delay_seconds=3.0),
        "queued": client.create_stress_run("queued", delay_seconds=0.1),
    }
    wait_for_queue_shape(
        client,
        concurrency=2,
        minimum_pending=1,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
        monotonic=monotonic,
    )
    expected = {"failed": "error", "long": "success", "queued": "success"}
    _wait_for_expected_statuses(
        client,
        runs,
        expected,
        concurrency=2,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
        monotonic=monotonic,
    )
    wait_for_queues_empty(
        client,
        concurrency=2,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
        monotonic=monotonic,
    )
    return runs


def seed_shutdown_probe(
    client: ProbeOperations,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, RunRef]:
    runs = {
        "long-a": client.create_stress_run("long-a", delay_seconds=6.0),
        "long-b": client.create_stress_run("long-b", delay_seconds=6.0),
        "queued": client.create_stress_run("queued", delay_seconds=0.1),
    }
    wait_for_queue_shape(
        client,
        concurrency=2,
        minimum_pending=1,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
        monotonic=monotonic,
    )
    validate_statuses(
        client,
        {name: runs[name] for name in ("long-a", "long-b")},
        {"long-a": "running", "long-b": "running"},
        concurrency=2,
    )
    return runs


def check_shutdown_probe(
    client: ProbeOperations,
    runs: Mapping[str, RunRef],
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    expected = {"long-a": "success", "long-b": "success", "queued": "success"}
    _wait_for_expected_statuses(
        client,
        runs,
        expected,
        concurrency=2,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
        monotonic=monotonic,
    )
    wait_for_queues_empty(
        client,
        concurrency=2,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
        monotonic=monotonic,
    )


def _serialize_runs(runs: Mapping[str, RunRef]) -> dict[str, dict[str, str]]:
    return {name: asdict(run) for name, run in runs.items()}


def _load_runs(path: Path) -> dict[str, RunRef]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), dict):
        raise SystemExit("State file must contain a JSON object with a runs object.")
    runs: dict[str, RunRef] = {}
    for name, value in payload["runs"].items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise SystemExit("Every state-file run must be a named JSON object.")
        thread_id = value.get("thread_id")
        run_id = value.get("run_id")
        if not isinstance(thread_id, str) or not isinstance(run_id, str):
            raise SystemExit(f"State-file run {name!r} must contain string thread_id and run_id values.")
        runs[name] = RunRef(thread_id=thread_id, run_id=run_id)
    return runs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--redis-url", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("bounded", "fanout", "failure", "shutdown-seed", "shutdown-check"),
    )
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be greater than zero")
    if args.mode == "shutdown-check" and args.state_file is None:
        parser.error("--state-file is required for --mode shutdown-check")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with ProbeClient(base_url=args.base_url, redis_url=args.redis_url) as client:
        if args.mode == "bounded":
            runs = run_bounded_probe(client, timeout_seconds=args.timeout_seconds)
        elif args.mode == "fanout":
            runs = run_fanout_probe(client, timeout_seconds=args.timeout_seconds)
        elif args.mode == "failure":
            runs = run_failure_probe(client, timeout_seconds=args.timeout_seconds)
        elif args.mode == "shutdown-seed":
            runs = seed_shutdown_probe(client, timeout_seconds=args.timeout_seconds)
        else:
            assert args.state_file is not None
            runs = _load_runs(args.state_file)
            check_shutdown_probe(client, runs, timeout_seconds=args.timeout_seconds)
        print(json.dumps({"mode": args.mode, "runs": _serialize_runs(runs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
