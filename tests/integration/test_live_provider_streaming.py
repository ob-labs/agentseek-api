import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.main import create_app
from agentseek_api.models.auth import User
from agentseek_api.services.run_jobs import RunExecutionJob
from agentseek_api.settings import settings


def _text_from_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if "content" in content:
            return _text_from_content(content["content"])
        return ""
    if isinstance(content, list):
        return "".join(_text_from_content(item) for item in content)
    return ""


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _parse_sse_events(stream_text: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for chunk in stream_text.strip().split("\n\n"):
        if not chunk.strip():
            continue
        event: dict[str, object] = {}
        for line in chunk.splitlines():
            if line.startswith("id: "):
                event["id"] = line.removeprefix("id: ")
            elif line.startswith("event: "):
                event["event"] = line.removeprefix("event: ")
            elif line.startswith("data: "):
                event["data"] = json.loads(line.removeprefix("data: "))
        if event:
            events.append(event)
    return events


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


async def header_user_override(request: Request) -> User:
    identity = request.headers.get("x-user-id", "live-provider-user")
    return User(identity=identity, is_authenticated=True)


def _provider_config() -> tuple[str, str, list[str]]:
    provider = os.getenv("LIVE_PROVIDER_KIND", "").strip().lower()
    if provider == "openai":
        return (
            "live_openai_stream",
            "LIVE_OPENAI_COMPAT_API_KEY",
            ["LIVE_OPENAI_COMPAT_MODEL", "LIVE_OPENAI_COMPAT_BASE_URL", "LIVE_OPENAI_COMPAT_API_KEY"],
        )
    if provider == "anthropic":
        return (
            "live_anthropic_stream",
            "LIVE_ANTHROPIC_COMPAT_API_KEY",
            ["LIVE_ANTHROPIC_COMPAT_MODEL", "LIVE_ANTHROPIC_COMPAT_BASE_URL", "LIVE_ANTHROPIC_COMPAT_API_KEY"],
        )
    pytest.skip("Set LIVE_PROVIDER_KIND to 'openai' or 'anthropic' to run live provider streaming checks.")


@pytest.fixture
def live_provider_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    graph_manifest = Path(__file__).resolve().parents[2] / "examples" / "live_provider_graphs" / "manifest.json"
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: InlineExecutor())
    monkeypatch.setattr("agentseek_api.services.langgraph_service._langgraph_service", None)
    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/live-provider.db")
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(graph_manifest))

    app = create_app()
    app.dependency_overrides[get_current_user] = header_user_override
    with TestClient(app) as test_client:
        yield test_client


def test_live_provider_stream_emits_multiple_message_chunks(live_provider_client: TestClient) -> None:
    graph_id, key_name, required_env = _provider_config()
    if not os.getenv(key_name, "").strip():
        pytest.skip(f"{key_name} is not configured for live provider streaming checks.")
    missing = [name for name in required_env if not os.getenv(name, "").strip()]
    assert not missing, f"Missing live provider configuration: {', '.join(missing)}"

    assistant = live_provider_client.post("/assistants", json={"name": f"{graph_id}-assistant", "graph_id": graph_id})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = live_provider_client.post("/threads", json={"metadata": {"suite": "live-provider-streaming"}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    run = live_provider_client.post(
        f"/threads/{thread_id}/runs",
        json={
            "assistant_id": assistant_id,
            "input": {
                "message": (
                    "Explain why token-level streaming verification matters in exactly two sentences, "
                    "using at least forty words and no bullet points."
                )
            },
        },
    )
    assert run.status_code == 200
    run_id = run.json()["run_id"]

    waited = live_provider_client.get(f"/threads/{thread_id}/runs/{run_id}/wait")
    assert waited.status_code == 200
    waited_body = waited.json()
    assert waited_body["status"] == "success", waited_body.get("last_error")
    assert waited_body["output"]["final_text"]

    fetched = live_provider_client.get(f"/threads/{thread_id}/runs/{run_id}")
    assert fetched.status_code == 200
    fetched_body = fetched.json()
    assert fetched_body["status"] == "success"
    assert fetched_body["output"]["final_text"] == waited_body["output"]["final_text"]

    stream = live_provider_client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
    assert stream.status_code == 200
    payloads = [
        json.loads(line.replace("data: ", "", 1))
        for line in stream.text.splitlines()
        if line.startswith("data: ")
    ]
    # The default run-stream replay returns the run's persisted protocol
    # events. A run created without an explicit stream_mode does not stream
    # incremental LLM tokens (no ``messages`` channel events), so the final
    # answer is read back from the last state snapshot (``values`` payload)
    # instead of token deltas.
    state_payloads = [
        payload
        for payload in payloads
        if isinstance(payload, dict) and isinstance(payload.get("messages"), list)
    ]
    assert state_payloads
    final_messages = state_payloads[-1]["messages"]
    final_ai = next(
        (m for m in reversed(final_messages) if isinstance(m, dict) and m.get("type") == "ai"),
        None,
    )
    assert final_ai is not None
    end_payloads = [payload for payload in payloads if payload.get("event") == "end"]

    assert payloads[0]["event"] == "start"
    assert "event: start" in stream.text
    assert "event: end" in stream.text
    assert end_payloads[-1]["status"] == "success"
    assert end_payloads[-1]["run_id"] == run_id
    assert _normalize_text(str(final_ai.get("content", ""))) == _normalize_text(
        waited_body["output"]["final_text"]
    )

    # Token-level proof: an explicit ``stream_mode=messages`` run must surface
    # real incremental ``messages/partial`` frames from the provider that
    # accumulate to the final answer. This restores the incremental token
    # assertion the manual live-provider workflow is contractually expected to
    # prove (the default replay above only proves the final snapshot).
    streamed = live_provider_client.post(
        f"/threads/{thread_id}/runs/stream",
        json={
            "assistant_id": assistant_id,
            "input": {
                "message": (
                    "Explain why token-level streaming verification matters in exactly two sentences, "
                    "using at least forty words and no bullet points."
                )
            },
            "stream_mode": "messages",
        },
    )
    assert streamed.status_code == 200, streamed.text
    assert streamed.headers["content-type"].startswith("text/event-stream")
    streamed_events = _parse_sse_events(streamed.text)
    streamed_names = [event["event"] for event in streamed_events]
    assert streamed_names[0] == "metadata"
    assert "messages/partial" in streamed_names
    assert not any("content-block" in name for name in streamed_names)

    partial_payloads = [event["data"] for event in streamed_events if event["event"] == "messages/partial"]
    assert len(partial_payloads) >= 2, "expected at least one incremental token step"
    accumulated_text = ""
    for payload in partial_payloads:
        if not isinstance(payload, list) or not payload:
            continue
        last = payload[-1]
        if not isinstance(last, dict) or last.get("type") != "ai":
            continue
        accumulated_text = _text_from_content(last.get("content"))
    assert accumulated_text, "messages/partial never carried an AI message"
    streamed_run_id = str(
        next(event["data"]["run_id"] for event in streamed_events if event["event"] == "metadata")
    )
    streamed_waited = live_provider_client.get(f"/threads/{thread_id}/runs/{streamed_run_id}/wait")
    assert streamed_waited.status_code == 200
    streamed_body = streamed_waited.json()
    assert streamed_body["status"] == "success", streamed_body.get("last_error")
    assert _normalize_text(accumulated_text) == _normalize_text(streamed_body["output"]["final_text"])
