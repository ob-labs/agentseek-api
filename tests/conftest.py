import asyncio
import json
import time
from pathlib import Path
from typing import Any
from collections.abc import Awaitable, Callable

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.main import create_app
from agentseek_api.models.auth import User
from agentseek_api.services.run_jobs import RunExecutionJob
from agentseek_api.settings import settings


class FakeCheckpointer:
    def __init__(self, connection_args: dict[str, str]) -> None:
        self.connection_args = connection_args

    def setup(self) -> None:
        return None

    def save_checkpoint(self, *, thread_id: str, run_id: str, payload: dict[str, Any]) -> None:
        _ = (thread_id, run_id, payload)

class InlineExecutor:
    async def submit(self, job: Callable[[], Awaitable[None]] | RunExecutionJob) -> None:
        if callable(job):
            await job()
            return
        from agentseek_api.services.run_preparation import _execute_and_persist

        await _execute_and_persist(
            run_id=job.run_id,
            thread_id=job.thread_id,
            user_id=job.user_id,
            payload=job.payload,
            graph_id=job.graph_id,
            kwargs=job.kwargs,
            resume=job.resume,
            is_resume=job.is_resume,
        )


class BackgroundInlineExecutor:
    """Submits run jobs as background tasks so a run stays genuinely active
    after the HTTP call that created it returns.

    The default ``InlineExecutor`` awaits the job inline, so ``POST /runs``
    returns only once the run is already terminal. That makes it structurally
    impossible to observe a run mid-flight. This executor is used by the
    mid-run reconnect regression tests to prove exactly-once semantics over a
    disconnect/reconnect on ``GET /runs/{id}/stream``.
    """

    async def submit(self, job: Callable[[], Awaitable[None]] | RunExecutionJob) -> None:
        if callable(job):
            asyncio.create_task(job())
            return
        from agentseek_api.services.run_preparation import _execute_and_persist

        asyncio.create_task(
            _execute_and_persist(
                run_id=job.run_id,
                thread_id=job.thread_id,
                user_id=job.user_id,
                payload=job.payload,
                graph_id=job.graph_id,
                kwargs=job.kwargs,
                resume=job.resume,
                is_resume=job.is_resume,
            )
        )


async def header_user_override(request: Request) -> User:
    identity = request.headers.get("x-user-id", "default_user")
    return User(identity=identity, is_authenticated=True)


async def _noop_ensure_default_assistants() -> None:
    return None


async def _collect_sse_frames(
    client,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    max_frames: int | None = None,
) -> list[tuple[int | None, str, dict[str, object]]]:
    """Collect SSE frames from a live stream, optionally stopping early to
    simulate a client disconnecting mid-run."""
    frames: list[tuple[int | None, str, dict[str, object]]] = []
    current_id: int | None = None
    current_event = ""
    current_data: list[str] = []
    async with client.stream("GET", url, headers=headers) as response:
        assert response.status_code == 200, response.text
        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line.startswith("id: "):
                    current_id = int(line[len("id: "):].strip())
                elif line.startswith("event: "):
                    current_event = line[len("event: "):].strip()
                elif line.startswith("data: "):
                    current_data.append(line[len("data: "):].strip())
                elif line == "" and current_data:
                    payload = json.loads("".join(current_data))
                    frames.append((current_id, current_event, payload))
                    current_id, current_event, current_data = None, "", []
                    if max_frames is not None and len(frames) >= max_frames:
                        return frames
    return frames


async def _wait_run_terminal(client, thread_id: str, run_id: str, *, timeout_seconds: float = 20.0) -> str:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        status = (await client.get(f"/threads/{thread_id}/runs/{run_id}")).json()["status"]
        if status in ("success", "error", "interrupted"):
            return status
        await asyncio.sleep(0.2)
    raise AssertionError(f"run {run_id} did not reach a terminal status within {timeout_seconds}s")


async def _midrun_reconnect_flow(app) -> None:
    """Connect to GET /runs/{id}/stream while the run is still executing,
    disconnect after a few frames, wait for the run to finish, then reconnect
    with Last-Event-ID and assert exactly-once delivery. Shared by the inline
    and Redis HTTP-level reconnect regression tests."""
    import httpx
    from httpx import ASGITransport

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            await _midrun_reconnect_checks(client)


async def _midrun_reconnect_checks(client) -> None:
    assistant = await client.post("/assistants", json={"name": "midrun", "graph_id": "stress_tool_agent"})
    assert assistant.status_code == 200, assistant.text
    assistant_id = assistant.json()["assistant_id"]

    thread = await client.post("/threads", json={"metadata": {"case": "midrun"}})
    assert thread.status_code == 200, thread.text
    thread_id = thread.json()["thread_id"]

    # ~4.5s total runtime (3 steps * 1.5s) guarantees a mid-run window.
    run = await client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": {"delay": 1.5, "steps": 3}},
    )
    assert run.status_code == 200, run.text
    run_id = run.json()["run_id"]

    stream_url = f"/threads/{thread_id}/runs/{run_id}/stream"

    # Phase 1: connect mid-run, collect a few frames, then disconnect.
    await asyncio.sleep(0.6)
    phase1 = await _collect_sse_frames(client, stream_url, max_frames=3)
    if not phase1:
        await asyncio.sleep(1.0)
        phase1 = await _collect_sse_frames(client, stream_url, max_frames=3)
    assert phase1, "expected to observe the run while it is still active"
    phase1_ids = [frame_id for frame_id, _, _ in phase1 if frame_id is not None]
    last_id = phase1_ids[-1]
    assert last_id is not None

    status = await _wait_run_terminal(client, thread_id, run_id)
    assert status == "success", status

    # Phase 2: reconnect with Last-Event-ID.
    phase2 = await _collect_sse_frames(client, stream_url, headers={"Last-Event-ID": str(last_id)})
    phase2_ids = [frame_id for frame_id, _, _ in phase2 if frame_id is not None]

    # Exactly-once: reconnect must not replay frames already delivered, and
    # must deliver every frame produced after the disconnect.
    assert all(frame_id > last_id for frame_id in phase2_ids), (
        f"reconnect replayed an already-delivered id: phase1={phase1_ids} phase2={phase2_ids}"
    )
    all_ids = phase1_ids + phase2_ids
    assert len(all_ids) == len(set(all_ids)), f"duplicate ids delivered across reconnect: {all_ids}"

    fresh = await _collect_sse_frames(client, stream_url)
    fresh_ids = [frame_id for frame_id, _, _ in fresh if frame_id is not None]
    assert [frame_id for frame_id in fresh_ids if frame_id > last_id] == phase2_ids, (
        f"reconnect content mismatch: full={fresh_ids} phase2={phase2_ids}"
    )


@pytest.fixture
def midrun_reconnect_flow() -> Callable[[object], Awaitable[None]]:
    """Async helper that drives a mid-run disconnect/reconnect against an app
    built by ``midrun_app_factory``, asserting exactly-once delivery."""
    return _midrun_reconnect_flow


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    from agentseek_api.core import auth_middleware

    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: InlineExecutor())
    monkeypatch.setattr("agentseek_api.main.ensure_default_assistants", _noop_ensure_default_assistants)
    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
    monkeypatch.setattr(
        "agentseek_api.core.auth_middleware.get_config_auth_settings",
        lambda: auth_middleware.ConfigAuthSettings(),
    )
    auth_middleware._backend = None

    app = create_app()
    app.dependency_overrides[get_current_user] = header_user_override
    with TestClient(app) as test_client:
        yield test_client
    auth_middleware._backend = None


@pytest.fixture
def midrun_app_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Callable[..., object]:
    """Build a fresh app whose runs execute in the background.

    Unlike the default ``client`` fixture (which awaits each run to completion
    before ``POST /runs`` returns), apps from this factory keep a run active
    after the create call, so a test can connect to ``GET /runs/{id}/stream``
    mid-flight, disconnect, and reconnect with ``Last-Event-ID``.

    ``executor_backend`` selects the run-stream storage path ("inline" or
    "redis"); ``redis_client`` (used only for the redis path) must be an object
    with the ``eval``/``xrange`` surface of the stream persistence layer.
    """
    from agentseek_api.core import auth_middleware

    def _make(*, executor_backend: str = "inline", redis_client: object | None = None) -> object:
        from agentseek_api.core import database as database_module
        from agentseek_api.services import stream_persistence as stream_module

        monkeypatch.setattr(database_module, "OceanBaseCheckpointSaver", FakeCheckpointer)
        monkeypatch.setattr("agentseek_api.main.ensure_default_assistants", _noop_ensure_default_assistants)
        monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/midrun-{executor_backend}.db")
        monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
        monkeypatch.setattr("agentseek_api.services.executor._executor", None)
        monkeypatch.setattr(settings, "EXECUTOR_BACKEND", executor_backend)
        monkeypatch.setattr(
            "agentseek_api.core.auth_middleware.get_config_auth_settings",
            lambda: auth_middleware.ConfigAuthSettings(),
        )
        auth_middleware._backend = None
        if redis_client is not None:
            monkeypatch.setattr(stream_module, "_redis_client", redis_client)
            from agentseek_api.api import runs as runs_api

            monkeypatch.setattr(runs_api, "REDIS_STREAM_POLL_INTERVAL_SECONDS", 0)

        monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: BackgroundInlineExecutor())

        app = create_app()
        app.dependency_overrides[get_current_user] = header_user_override
        return app

    return _make
