# A2A Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add strict, LangSmith-style A2A support to `agentseek-api` with `/a2a/{assistant_id}`, `/.well-known/agent-card.json`, assistant-first metadata, `message/send`, `message/stream`, `tasks/get`, `tasks/cancel`, and verified client interoperability.

**Architecture:** Reuse the existing assistant table, graph registry, auth dependency flow, and runtime config wiring, but isolate the A2A protocol adaptation in a dedicated `a2a_server.py` module. Enable the feature at startup through a fail-closed `http.disable_a2a` gate, keep task tracking in-process for parity, and expose only message-compatible assistants through the protocol.

**Tech Stack:** FastAPI, Starlette SSE responses, Pydantic, SQLAlchemy ORM, LangGraph, httpx, pytest, pytest-asyncio, uvicorn, `a2a-sdk`

---

## File Structure

- Create: `src/agentseek_api/core/a2a_config.py`
  - single-purpose fail-closed startup toggle for `http.disable_a2a`
- Create: `src/agentseek_api/a2a_server.py`
  - assistant lookup, compatibility checks, Agent Card shaping, JSON-RPC method dispatch, in-memory task registry, SSE streaming helpers
- Modify: `src/agentseek_api/main.py`
  - startup-time enablement, route registration, feature flag reporting
- Modify: `pyproject.toml`
  - add `a2a-sdk` to dev dependencies for live interoperability coverage
- Modify: `uv.lock`
  - lockfile update for `a2a-sdk`
- Test: `tests/unit/test_a2a_config.py`
  - config gate behavior
- Test: `tests/unit/test_a2a_server.py`
  - compatibility helpers, Agent Card metadata shaping, JSON-RPC/task helper behavior
- Test: `tests/integration/test_a2a_endpoint.py`
  - auth, `message/send`, `message/stream`, `tasks/get`, `tasks/cancel`, disabled route handling, incompatible assistants
- Test: `tests/integration/test_agent_card.py`
  - discovery behavior and assistant-first metadata
- Modify: `tests/integration/test_system_endpoints.py`
  - `flags.a2a` assertions
- Test: `tests/e2e/test_a2a_live.py`
  - live server + A2A SDK client proof
- Modify: `README.md`
  - shipped A2A behavior only

### Task 1: Add fail-closed A2A config gating

**Files:**
- Create: `tests/unit/test_a2a_config.py`
- Create: `src/agentseek_api/core/a2a_config.py`
- Modify: `src/agentseek_api/main.py`

- [ ] **Step 1: Write the failing config tests**

```python
from pathlib import Path

from agentseek_api.core.a2a_config import is_a2a_enabled
from agentseek_api.core.config_file import get_active_config_payload
from agentseek_api.settings import settings


def test_is_a2a_enabled_defaults_true_without_http_section(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text('{"graphs":{"chat":"chat.graph:graph"}}', encoding="utf-8")
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    assert is_a2a_enabled() is True


def test_is_a2a_enabled_respects_disable_flag(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        """
{
  "graphs": {"chat": "chat.graph:graph"},
  "http": {"disable_a2a": true}
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    assert is_a2a_enabled() is False


def test_is_a2a_enabled_fails_closed_for_invalid_config(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        """
{
  "graphs": {"chat": "chat.graph:graph"},
  "http": {"disable_a2a": true}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    assert get_active_config_payload() is None
    assert is_a2a_enabled() is False


def test_is_a2a_enabled_fails_closed_for_invalid_http_section(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        """
{
  "graphs": {"chat": "chat.graph:graph"},
  "http": []
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    assert is_a2a_enabled() is False


def test_is_a2a_enabled_fails_closed_for_invalid_disable_flag(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        """
{
  "graphs": {"chat": "chat.graph:graph"},
  "http": {"disable_a2a": "true"}
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    assert is_a2a_enabled() is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_a2a_config.py -q`
Expected: FAIL with `ModuleNotFoundError` for `agentseek_api.core.a2a_config` or missing `is_a2a_enabled`

- [ ] **Step 3: Write the minimal config gate and wire the startup flag**

```python
# src/agentseek_api/core/a2a_config.py
from __future__ import annotations

from agentseek_api.core.config_file import active_config_path, get_active_config_payload


def is_a2a_enabled() -> bool:
    config_path = active_config_path()
    if config_path is None:
        return True
    payload = get_active_config_payload()
    if payload is None:
        return False
    if "http" not in payload:
        return True
    http = payload.get("http")
    if not isinstance(http, dict):
        return False
    disable_a2a = http.get("disable_a2a")
    if disable_a2a is None:
        return True
    if isinstance(disable_a2a, bool):
        return disable_a2a is not True
    return False
```

```python
# src/agentseek_api/main.py
from agentseek_api.core.a2a_config import is_a2a_enabled


def _feature_flags(*, a2a_enabled: bool, mcp_enabled: bool) -> dict[str, bool]:
    return {
        "agents": True,
        "assistants": True,
        "threads": True,
        "runs": True,
        "crons": False,
        "store": True,
        "a2a": a2a_enabled,
        "mcp": mcp_enabled,
        "protocol_v2": True,
    }


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, version=__version__, lifespan=lifespan)
    _apply_auth_openapi(app)
    app.state.a2a_enabled = is_a2a_enabled()
    app.state.mcp_enabled = is_mcp_enabled()
```

- [ ] **Step 4: Run the config and system tests**

Run: `uv run pytest tests/unit/test_a2a_config.py tests/integration/test_system_endpoints.py -q`
Expected: PASS for the new unit file and FAIL in `test_info_endpoint` until `flags.a2a` assertions are updated in a later task

- [ ] **Step 5: Commit the config gate**

```bash
git add src/agentseek_api/core/a2a_config.py src/agentseek_api/main.py tests/unit/test_a2a_config.py
git commit -m "feat: add A2A startup config gate"
```

### Task 2: Add compatibility helpers, Agent Card shaping, and route registration

**Files:**
- Create: `tests/unit/test_a2a_server.py`
- Create: `tests/integration/test_agent_card.py`
- Create: `src/agentseek_api/a2a_server.py`
- Modify: `src/agentseek_api/main.py`
- Modify: `tests/integration/test_system_endpoints.py`

- [ ] **Step 1: Write the failing helper and Agent Card tests**

```python
from agentseek_api.a2a_server import build_agent_card, is_a2a_compatible_entry
from agentseek_api.models.api import AssistantRead
from agentseek_api.services.langgraph_service import GraphEntry


def test_is_a2a_compatible_entry_accepts_messages_schema() -> None:
    entry = GraphEntry(
        graph_factory=lambda **_: None,
        prepare_input=lambda payload: payload,
        extract_output=lambda result, payload: payload,
        tool_name="chat",
        description="Chat graph",
        input_schema={
            "type": "object",
            "properties": {"messages": {"type": "array"}},
            "required": ["messages"],
        },
        output_schema={"type": "object"},
    )

    assert is_a2a_compatible_entry(entry) is True


def test_is_a2a_compatible_entry_rejects_non_message_schema() -> None:
    entry = GraphEntry(
        graph_factory=lambda **_: None,
        prepare_input=lambda payload: payload,
        extract_output=lambda result, payload: payload,
        tool_name="echo",
        input_schema={
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
        output_schema={"type": "object"},
    )

    assert is_a2a_compatible_entry(entry) is False


def test_build_agent_card_prefers_assistant_metadata() -> None:
    assistant = AssistantRead.model_construct(
        assistant_id="assistant-1",
        name="Support Agent",
        graph_id="chat_graph",
        created_at=None,
        updated_at=None,
        metadata={},
        config={},
        context={},
        version=1,
        description="Handles support requests",
    )
    entry = GraphEntry(
        graph_factory=lambda **_: None,
        prepare_input=lambda payload: payload,
        extract_output=lambda result, payload: payload,
        tool_name="chat_graph",
        description="Graph description should not override assistant description",
        input_schema={
            "type": "object",
            "properties": {"messages": {"type": "array"}},
            "required": ["messages"],
        },
        output_schema={"type": "object"},
    )

    card = build_agent_card(
        base_url="http://127.0.0.1:2024",
        assistant=assistant,
        entry=entry,
    )

    assert card["name"] == "Support Agent"
    assert card["description"] == "Handles support requests"
    assert card["url"] == "http://127.0.0.1:2024/a2a/assistant-1"
```

```python
from fastapi.testclient import TestClient


def test_agent_card_returns_assistant_first_metadata(auth_client: TestClient) -> None:
    create_response = auth_client.post(
        "/assistants",
        headers={"X-API-Key": "secret"},
        json={"name": "Support Agent", "graph_id": "stress_test", "description": "Chat assistant"},
    )
    assistant_id = create_response.json()["assistant_id"]

    response = auth_client.get(
        f"/.well-known/agent-card.json?assistant_id={assistant_id}",
        headers={"X-API-Key": "secret"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Support Agent"
    assert body["description"] == "Chat assistant"
    assert body["url"].endswith(f"/a2a/{assistant_id}")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_a2a_server.py tests/integration/test_agent_card.py -q`
Expected: FAIL because `agentseek_api.a2a_server` does not exist and the Agent Card route is not mounted

- [ ] **Step 3: Add the helper module and mount the discovery route**

```python
# src/agentseek_api/a2a_server.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Assistant
from agentseek_api.models.api import AssistantRead
from agentseek_api.services.langgraph_service import GraphEntry, LangGraphService, get_langgraph_service


def is_a2a_compatible_entry(entry: GraphEntry) -> bool:
    properties = entry.input_schema.get("properties")
    required = entry.input_schema.get("required", [])
    if not isinstance(properties, dict):
        return False
    messages = properties.get("messages")
    if not isinstance(messages, dict):
        return False
    return messages.get("type") == "array" and "messages" in required


def build_agent_card(*, base_url: str, assistant: AssistantRead, entry: GraphEntry) -> dict[str, Any]:
    return {
        "name": assistant.name,
        "description": assistant.description or "",
        "url": f"{base_url}/a2a/{assistant.assistant_id}",
        "preferredTransport": "JSONRPC",
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": entry.tool_name,
                "name": assistant.name,
                "description": assistant.description or entry.description,
            }
        ],
    }


async def load_assistant(assistant_id: str) -> AssistantRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = await session.scalar(select(Assistant).where(Assistant.assistant_id == assistant_id))
        if row is None:
            raise HTTPException(status_code=404, detail="Assistant not found")
        return AssistantRead(
            assistant_id=row.assistant_id,
            name=row.name,
            graph_id=row.graph_id,
            created_at=row.created_at,
            updated_at=row.updated_at,
            metadata=row.metadata_json,
            config=row.config_json,
            context=row.context_json,
            version=row.version,
            description=row.description,
        )
```

```python
# src/agentseek_api/main.py
from fastapi import Depends, Request

from agentseek_api.a2a_server import build_agent_card, is_a2a_compatible_entry, load_assistant


@app.get("/.well-known/agent-card.json", include_in_schema=False)
async def agent_card(
    request: Request,
    assistant_id: str,
    user=Depends(get_current_user),
) -> dict[str, object]:
    _ = user
    assistant = await load_assistant(assistant_id)
    entry = get_langgraph_service().get_entry(assistant.graph_id)
    if not is_a2a_compatible_entry(entry):
        raise HTTPException(status_code=400, detail="Assistant graph is not A2A-compatible")
    return build_agent_card(
        base_url=str(request.base_url).rstrip("/"),
        assistant=assistant,
        entry=entry,
    )
```

- [ ] **Step 4: Run the helper, Agent Card, and system tests**

Run: `uv run pytest tests/unit/test_a2a_server.py tests/integration/test_agent_card.py tests/integration/test_system_endpoints.py -q`
Expected: PASS for helper and Agent Card tests, FAIL in system endpoint assertions until `flags.a2a` is updated where expected

- [ ] **Step 5: Commit the helper and discovery surface**

```bash
git add src/agentseek_api/a2a_server.py src/agentseek_api/main.py tests/unit/test_a2a_server.py tests/integration/test_agent_card.py tests/integration/test_system_endpoints.py
git commit -m "feat: add A2A compatibility helpers and agent card"
```

### Task 3: Implement `message/send` and `tasks/get`

**Files:**
- Modify: `tests/unit/test_a2a_server.py`
- Create: `tests/integration/test_a2a_endpoint.py`
- Modify: `src/agentseek_api/a2a_server.py`
- Modify: `src/agentseek_api/main.py`

- [ ] **Step 1: Write the failing `message/send` and `tasks/get` tests**

```python
def test_message_send_returns_completed_task(auth_client: TestClient) -> None:
    assistant = auth_client.post(
        "/assistants",
        headers={"X-API-Key": "secret"},
        json={"name": "Messages Echo", "graph_id": "stress_test", "description": "Echoes text"},
    ).json()

    response = auth_client.post(
        f"/a2a/{assistant['assistant_id']}",
        headers={"X-API-Key": "secret", "Accept": "application/json"},
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "{\"delay\":0.0,\"steps\":1,\"note\":\"hello from a2a\"}"}],
                    "messageId": "msg-1",
                }
            },
        },
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["kind"] == "task"
    assert result["status"]["state"] == "completed"
    assert result["contextId"]
    assert "hello from a2a" in result["artifacts"][0]["parts"][0]["text"]


def test_tasks_get_returns_saved_snapshot(auth_client: TestClient) -> None:
    assistant = auth_client.post(
        "/assistants",
        headers={"X-API-Key": "secret"},
        json={"name": "Messages Echo", "graph_id": "stress_test", "description": "Echoes text"},
    ).json()
    send_response = auth_client.post(
        f"/a2a/{assistant['assistant_id']}",
        headers={"X-API-Key": "secret", "Accept": "application/json"},
        json={
            "jsonrpc": "2.0",
            "id": "2",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "{\"delay\":0.0,\"steps\":1,\"note\":\"lookup me later\"}"}],
                    "messageId": "msg-2",
                }
            },
        },
    )
    task_id = send_response.json()["result"]["id"]

    get_response = auth_client.post(
        f"/a2a/{assistant['assistant_id']}",
        headers={"X-API-Key": "secret", "Accept": "application/json"},
        json={
            "jsonrpc": "2.0",
            "id": "3",
            "method": "tasks/get",
            "params": {"id": task_id},
        },
    )

    assert get_response.status_code == 200
    assert get_response.json()["result"]["id"] == task_id
    assert get_response.json()["result"]["status"]["state"] == "completed"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/integration/test_a2a_endpoint.py -q -k "message_send or tasks_get"`
Expected: FAIL with 404 on `/a2a/{assistant_id}` or JSON-RPC method-not-found behavior

- [ ] **Step 3: Implement minimal JSON-RPC dispatch, task registry, and synchronous execution**

```python
# src/agentseek_api/a2a_server.py
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from langgraph.constants import CONF, CONFIG_KEY_CHECKPOINTER

from agentseek_api.core.runtime_store import UserScopedStore
from agentseek_api.models.auth import User


@dataclass
class A2ATaskRecord:
    task_id: str
    assistant_id: str
    context_id: str
    state: str = "submitted"
    message: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)


class A2ATaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, A2ATaskRecord] = {}

    def save(self, record: A2ATaskRecord) -> None:
        self._tasks[record.task_id] = record

    def get(self, task_id: str) -> A2ATaskRecord:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise ValueError(f"Unknown task: {task_id}") from exc


def build_graph_config(*, user: User, context_id: str) -> tuple[UserScopedStore, dict[str, Any]]:
    runtime_store = UserScopedStore(db_manager.get_store(), user_id=user.identity)
    checkpointer = db_manager.get_langgraph_checkpointer()
    return runtime_store, {
        CONF: {
            "thread_id": context_id,
            "checkpoint_ns": f"a2a:{uuid4()}",
            CONFIG_KEY_CHECKPOINTER: checkpointer,
            "store": runtime_store,
            "langgraph_auth_user": user.model_dump(),
        }
    }


def make_text_artifact(text: str) -> dict[str, Any]:
    return {
        "artifactId": str(uuid4()),
        "name": "Assistant Response",
        "parts": [{"kind": "text", "text": text}],
    }
```

```python
# src/agentseek_api/main.py
from fastapi import Depends


@app.post("/a2a/{assistant_id}", include_in_schema=False)
async def a2a_jsonrpc(
    assistant_id: str,
    payload: dict[str, object],
    user=Depends(get_current_user),
) -> dict[str, object]:
    return await handle_a2a_request(
        assistant_id=assistant_id,
        payload=payload,
        user=user,
        service=get_langgraph_service(),
        registry=app.state.a2a_registry,
    )


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, version=__version__, lifespan=lifespan)
    _apply_auth_openapi(app)
    app.state.a2a_enabled = is_a2a_enabled()
    app.state.a2a_registry = A2ATaskRegistry()
    app.state.mcp_enabled = is_mcp_enabled()
```

- [ ] **Step 4: Run the targeted A2A tests**

Run: `uv run pytest tests/integration/test_a2a_endpoint.py -q -k "message_send or tasks_get"`
Expected: PASS with completed task responses and stable task lookup

- [ ] **Step 5: Commit send/get support**

```bash
git add src/agentseek_api/a2a_server.py src/agentseek_api/main.py tests/integration/test_a2a_endpoint.py tests/unit/test_a2a_server.py
git commit -m "feat: implement A2A message send and task lookup"
```

### Task 4: Implement `message/stream`, `tasks/cancel`, and strict error behavior

**Files:**
- Modify: `tests/integration/test_a2a_endpoint.py`
- Modify: `tests/unit/test_a2a_server.py`
- Modify: `src/agentseek_api/a2a_server.py`
- Modify: `src/agentseek_api/main.py`

- [ ] **Step 1: Write the failing stream, cancel, and protocol-error tests**

```python
def test_message_stream_returns_sse_events(auth_client: TestClient) -> None:
    assistant = auth_client.post(
        "/assistants",
        headers={"X-API-Key": "secret"},
        json={"name": "Messages Echo", "graph_id": "stress_test", "description": "Echoes text"},
    ).json()

    with auth_client.stream(
        "POST",
        f"/a2a/{assistant['assistant_id']}",
        headers={"X-API-Key": "secret", "Accept": "text/event-stream"},
        json={
            "jsonrpc": "2.0",
            "id": "10",
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "{\"delay\":0.0,\"steps\":1,\"note\":\"stream me\"}"}],
                    "messageId": "msg-stream",
                }
            },
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode("utf-8")

    assert response.status_code == 200
    assert "event:" in body
    assert '"state": "completed"' in body
    assert 'stream me' in body


def test_tasks_cancel_returns_cancelled_or_terminal_state(auth_client: TestClient) -> None:
    assistant = auth_client.post(
        "/assistants",
        headers={"X-API-Key": "secret"},
        json={"name": "Messages Echo", "graph_id": "stress_test", "description": "Echoes text"},
    ).json()
    send_response = auth_client.post(
        f"/a2a/{assistant['assistant_id']}",
        headers={"X-API-Key": "secret"},
        json={
            "jsonrpc": "2.0",
            "id": "11",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "{\"delay\":0.0,\"steps\":1,\"note\":\"cancel me\"}"}],
                    "messageId": "msg-cancel",
                }
            },
        },
    )
    task_id = send_response.json()["result"]["id"]

    cancel_response = auth_client.post(
        f"/a2a/{assistant['assistant_id']}",
        headers={"X-API-Key": "secret"},
        json={
            "jsonrpc": "2.0",
            "id": "12",
            "method": "tasks/cancel",
            "params": {"id": task_id},
        },
    )

    assert cancel_response.status_code == 200
    assert cancel_response.json()["result"]["id"] == task_id
    assert cancel_response.json()["result"]["status"]["state"] in {"cancelled", "completed"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/integration/test_a2a_endpoint.py -q -k "message_stream or tasks_cancel"`
Expected: FAIL because streaming and cancellation have not been implemented

- [ ] **Step 3: Add SSE response shaping, cancel handling, and JSON-RPC error helpers**

```python
# src/agentseek_api/a2a_server.py
import json

from fastapi.responses import StreamingResponse


def jsonrpc_error(*, request_id: object, code: int, message: str) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def jsonrpc_result(*, request_id: object, result: dict[str, Any]) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


async def stream_task_result(*, request_id: object, result: dict[str, Any]) -> StreamingResponse:
    async def iterator() -> AsyncIterator[str]:
        yield f"event: message\n"
        yield f"data: {json.dumps(jsonrpc_result(request_id=request_id, result=result))}\n\n"

    return StreamingResponse(iterator(), media_type="text/event-stream")


def cancel_task(registry: A2ATaskRegistry, task_id: str) -> A2ATaskRecord:
    record = registry.get(task_id)
    if record.state not in {"completed", "failed", "cancelled"}:
        record.state = "cancelled"
        record.message = "Task cancelled"
    registry.save(record)
    return record
```

- [ ] **Step 4: Run the streaming, cancel, and full A2A integration suite**

Run: `uv run pytest tests/integration/test_a2a_endpoint.py tests/unit/test_a2a_server.py -q`
Expected: PASS with SSE framing, terminal task state, and stable JSON-RPC errors

- [ ] **Step 5: Commit stream and cancel support**

```bash
git add src/agentseek_api/a2a_server.py src/agentseek_api/main.py tests/integration/test_a2a_endpoint.py tests/unit/test_a2a_server.py
git commit -m "feat: add A2A streaming and task cancellation"
```

### Task 5: Add live A2A interoperability coverage

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `tests/e2e/test_a2a_live.py`

- [ ] **Step 1: Write the failing live interoperability test**

```python
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Generator

import httpx
import pytest
import uvicorn
from a2a.client.client_factory import ClientFactory

from agentseek_api.core import auth_middleware
from agentseek_api.main import create_app
from agentseek_api.services import langgraph_service as langgraph_service_module
from agentseek_api.settings import settings


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_a2a_client_can_fetch_card_send_and_get_task(a2a_live_base_url: str) -> None:
    async with httpx.AsyncClient(
        headers={"x-user-id": "a2a-e2e-user"},
        timeout=10.0,
        trust_env=False,
    ) as http_client:
        card_response = await http_client.get(f"{a2a_live_base_url}/.well-known/agent-card.json", params={"assistant_id": "assistant-under-test"})
        assert card_response.status_code == 200

        client = await ClientFactory.create_from_url(
            card_response.json()["url"],
            httpx_client=http_client,
        )
        task = await client.send_message(
            {
                "role": "user",
                "parts": [{"kind": "text", "text": "hello from a2a sdk"}],
                "messageId": "live-msg-1",
            }
        )
        assert task.status.state == "completed"

        fetched = await client.get_task({"id": task.id})
        assert fetched.id == task.id
```

- [ ] **Step 2: Run the live test to verify it fails**

Run: `uv run pytest tests/e2e/test_a2a_live.py -q`
Expected: FAIL because `a2a-sdk` is not installed and the live A2A server surface is incomplete

- [ ] **Step 3: Add the SDK dependency and adapt the live fixture**

```toml
# pyproject.toml
[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "httpx>=0.27.0",
    "ruff>=0.6.0",
    "a2a-sdk>=1.0.3",
]
```

```python
# tests/e2e/test_a2a_live.py
assistant_response = await http_client.post(
    f"{a2a_live_base_url}/assistants",
    headers={"x-user-id": "a2a-e2e-user"},
    json={"name": "Live A2A", "graph_id": "stress_test", "description": "Live A2A test assistant"},
)
assistant_id = assistant_response.json()["assistant_id"]

card_response = await http_client.get(
    f"{a2a_live_base_url}/.well-known/agent-card.json",
    params={"assistant_id": assistant_id},
)
```

- [ ] **Step 4: Run the live A2A test**

Run: `uv sync`
Expected: dependency installation succeeds

Run: `uv run pytest tests/e2e/test_a2a_live.py -q`
Expected: PASS with a real A2A SDK client fetching the Agent Card and calling the endpoint

- [ ] **Step 5: Commit the live interoperability coverage**

```bash
git add pyproject.toml uv.lock tests/e2e/test_a2a_live.py
git commit -m "test: verify A2A client interoperability"
```

### Task 6: Update README and run the final verification slice

**Files:**
- Modify: `README.md`
- Modify: `tests/integration/test_system_endpoints.py`

- [ ] **Step 1: Write the README and flag assertions that should fail first**

```python
def test_info_endpoint(client: TestClient) -> None:
    response = client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert body["flags"]["a2a"] is True
    assert body["flags"]["mcp"] is True


def test_metrics_endpoint_json_format(client: TestClient) -> None:
    response = client.get("/metrics?format=json")
    assert response.status_code == 200
    body = response.json()
    assert body["flags"]["a2a"] is True
```

```md
## A2A

AgentSeek API exposes assistants through a LangSmith-style A2A JSON-RPC endpoint at `/a2a/{assistant_id}` and an Agent Card discovery endpoint at `/.well-known/agent-card.json?assistant_id={assistant_id}`.

Supported methods:

- `message/send`
- `message/stream`
- `tasks/get`
- `tasks/cancel`

A2A is enabled by default. To disable it, set `http.disable_a2a` to `true` in `agentseek.json` or `langgraph.json`.

Agent Cards use assistant-first metadata:

- `name` and `description` come from the assistant row
- capabilities are derived from the assistant's bound graph

Only assistants backed by message-compatible graphs are exposed through A2A.
```

- [ ] **Step 2: Run the final suite before README implementation is complete**

Run: `uv run pytest tests/unit/test_a2a_config.py tests/unit/test_a2a_server.py tests/integration/test_agent_card.py tests/integration/test_a2a_endpoint.py tests/integration/test_system_endpoints.py tests/e2e/test_a2a_live.py -q`
Expected: FAIL until `flags.a2a` is wired everywhere and README changes are committed

- [ ] **Step 3: Finalize README text and complete endpoint flag assertions**

```python
# tests/integration/test_system_endpoints.py
assert body["flags"]["a2a"] is True
assert response.json()["flags"]["a2a"] is True
assert body["flags"]["mcp"] is True
```

```md
### Disable A2A

~~~json
{
  "$schema": "https://langgra.ph/schema.json",
  "http": {
    "disable_a2a": true
  }
}
~~~
```

- [ ] **Step 4: Run the full verification slice**

Run: `uv run pytest tests/unit/test_a2a_config.py tests/unit/test_a2a_server.py tests/integration/test_agent_card.py tests/integration/test_a2a_endpoint.py tests/integration/test_system_endpoints.py tests/e2e/test_a2a_live.py -q`
Expected: PASS

Run: `uv run ruff check src tests`
Expected: PASS

- [ ] **Step 5: Commit the docs and final verification state**

```bash
git add README.md tests/integration/test_system_endpoints.py
git commit -m "docs: document A2A endpoint usage"
```
