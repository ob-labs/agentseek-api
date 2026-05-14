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
from agentseek_api.settings import settings


class FakeCheckpointer:
    def __init__(self, connection_args: dict[str, str]) -> None:
        self.connection_args = connection_args

    def setup(self) -> None:
        return None

    def save_checkpoint(self, *, thread_id: str, run_id: str, payload: dict[str, Any]) -> None:
        _ = (thread_id, run_id, payload)


class InlineExecutor:
    async def submit(self, func: Callable[[], Awaitable[None]]) -> None:
        await func()


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
    assert waited_body["status"] == "success"
    assert waited_body["output"]["final_text"]

    stream = live_provider_client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
    assert stream.status_code == 200
    payloads = [
        json.loads(line.replace("data: ", "", 1))
        for line in stream.text.splitlines()
        if line.startswith("data: ")
    ]
    message_chunks = [
        payload
        for payload in payloads
        if payload.get("event") == "message_chunk"
        and payload.get("langgraph_event") in {"on_chat_model_stream", "on_llm_stream"}
        and str(payload.get("content", "")).strip()
    ]

    assert "event: start" in stream.text
    assert "event: end" in stream.text
    assert len(message_chunks) >= 2
