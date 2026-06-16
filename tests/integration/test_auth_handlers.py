"""Integration tests with real auth handlers enabled.

Verifies that authorize() correctly:
- blocks forbidden operations (403)
- injects metadata on create
- filters resources by owner on search/read
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph_sdk import Auth

from agentseek_api.core import auth_middleware
from agentseek_api.core.auth_middleware import LangGraphAuthBackend
from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.main import create_app
from agentseek_api.models.auth import User
from agentseek_api.settings import settings


def _build_auth_object() -> Auth:
    auth = Auth()

    @auth.authenticate
    async def authenticate(authorization: str | None):
        if not authorization:
            raise Auth.exceptions.HTTPException(status_code=401, detail="Missing token")
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            raise Auth.exceptions.HTTPException(status_code=401, detail="Bad scheme")
        users = {"alice-token": "alice", "bob-token": "bob"}
        identity = users.get(token)
        if identity is None:
            raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token")
        return {"identity": identity, "is_authenticated": True}

    @auth.on
    async def add_owner(ctx, value):
        if ctx.action == "delete":
            raise Auth.exceptions.HTTPException(status_code=403, detail="Deletion not allowed")
        metadata = value.setdefault("metadata", {})
        metadata["owner"] = ctx.user.identity
        return {"owner": ctx.user.identity}

    return auth


@pytest.fixture
def auth_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    from tests.conftest import FakeCheckpointer, InlineExecutor, _noop_ensure_default_assistants

    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: InlineExecutor())
    monkeypatch.setattr("agentseek_api.main.ensure_default_assistants", _noop_ensure_default_assistants)
    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
    monkeypatch.setattr(
        "agentseek_api.core.auth_middleware.get_config_auth_settings",
        lambda: auth_middleware.ConfigAuthSettings(),
    )

    backend = LangGraphAuthBackend(_build_auth_object())
    auth_middleware._backend = backend

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    auth_middleware._backend = None


ALICE_HEADERS = {"Authorization": "Bearer alice-token"}
BOB_HEADERS = {"Authorization": "Bearer bob-token"}


def test_unauthenticated_request_returns_401(auth_client: TestClient) -> None:
    resp = auth_client.post("/threads", json={})
    assert resp.status_code == 401


def test_invalid_token_returns_401(auth_client: TestClient) -> None:
    resp = auth_client.post("/threads", json={}, headers={"Authorization": "Bearer bad-token"})
    assert resp.status_code == 401


def test_create_thread_injects_owner_metadata(auth_client: TestClient) -> None:
    resp = auth_client.post("/threads", json={}, headers=ALICE_HEADERS)
    assert resp.status_code == 200
    thread = resp.json()
    assert thread["metadata"]["owner"] == "alice"


def test_user_cannot_see_other_users_threads(auth_client: TestClient) -> None:
    alice_resp = auth_client.post("/threads", json={}, headers=ALICE_HEADERS)
    assert alice_resp.status_code == 200
    thread_id = alice_resp.json()["thread_id"]

    bob_get = auth_client.get(f"/threads/{thread_id}", headers=BOB_HEADERS)
    assert bob_get.status_code == 404


def test_search_threads_filters_by_owner(auth_client: TestClient) -> None:
    auth_client.post("/threads", json={"metadata": {"label": "a1"}}, headers=ALICE_HEADERS)
    auth_client.post("/threads", json={"metadata": {"label": "b1"}}, headers=BOB_HEADERS)

    alice_search = auth_client.post("/threads/search", json={}, headers=ALICE_HEADERS)
    assert alice_search.status_code == 200
    alice_threads = alice_search.json()
    assert all(t["metadata"].get("owner") == "alice" for t in alice_threads)

    bob_search = auth_client.post("/threads/search", json={}, headers=BOB_HEADERS)
    assert bob_search.status_code == 200
    bob_threads = bob_search.json()
    assert all(t["metadata"].get("owner") == "bob" for t in bob_threads)


def test_create_assistant_injects_owner_metadata(auth_client: TestClient) -> None:
    resp = auth_client.post(
        "/assistants",
        json={"name": "test-assistant", "graph_id": "default"},
        headers=ALICE_HEADERS,
    )
    assert resp.status_code == 200
    assistant = resp.json()
    assert assistant["metadata"]["owner"] == "alice"


def test_search_assistants_filters_by_owner(auth_client: TestClient) -> None:
    auth_client.post(
        "/assistants",
        json={"name": "alice-asst", "graph_id": "default"},
        headers=ALICE_HEADERS,
    )
    auth_client.post(
        "/assistants",
        json={"name": "bob-asst", "graph_id": "default"},
        headers=BOB_HEADERS,
    )

    alice_search = auth_client.post("/assistants/search", json={}, headers=ALICE_HEADERS)
    assert alice_search.status_code == 200
    assert all(a["metadata"].get("owner") == "alice" for a in alice_search.json())

    bob_search = auth_client.post("/assistants/search", json={}, headers=BOB_HEADERS)
    assert bob_search.status_code == 200
    assert all(a["metadata"].get("owner") == "bob" for a in bob_search.json())


def test_cron_operations_filter_by_owner(auth_client: TestClient) -> None:
    asst = auth_client.post(
        "/assistants",
        json={"name": "cron-asst", "graph_id": "default"},
        headers=ALICE_HEADERS,
    )
    assistant_id = asst.json()["assistant_id"]

    alice_cron = auth_client.post(
        "/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=MINUTELY;INTERVAL=5", "input": {}},
        headers=ALICE_HEADERS,
    )
    assert alice_cron.status_code == 200
    cron_id = alice_cron.json()["cron_id"]

    bob_get = auth_client.get(f"/runs/crons/{cron_id}", headers=BOB_HEADERS)
    assert bob_get.status_code == 404

    bob_search = auth_client.post("/runs/crons/search", json={}, headers=BOB_HEADERS)
    assert bob_search.status_code == 200
    assert not any(c["cron_id"] == cron_id for c in bob_search.json()["items"])

    # The select-projection branch returns a raw JSONResponse, bypassing the
    # response_model — confirm it still applies owner filters (no leak).
    bob_select = auth_client.post(
        "/runs/crons/search",
        json={"select": ["cron_id"]},
        headers=BOB_HEADERS,
    )
    assert bob_select.status_code == 200
    assert not any(c.get("cron_id") == cron_id for c in bob_select.json()["items"])

    alice_select = auth_client.post(
        "/runs/crons/search",
        json={"select": ["cron_id"]},
        headers=ALICE_HEADERS,
    )
    assert alice_select.status_code == 200
    assert any(c.get("cron_id") == cron_id for c in alice_select.json()["items"])

    alice_get = auth_client.get(f"/runs/crons/{cron_id}", headers=ALICE_HEADERS)
    assert alice_get.status_code == 200


def test_delete_thread_returns_403_when_handler_rejects(auth_client: TestClient) -> None:
    resp = auth_client.post("/threads", json={}, headers=ALICE_HEADERS)
    assert resp.status_code == 200
    thread_id = resp.json()["thread_id"]

    delete_resp = auth_client.delete(f"/threads/{thread_id}", headers=ALICE_HEADERS)
    assert delete_resp.status_code == 403
    assert "not allowed" in delete_resp.json()["detail"].lower()


def test_user_model_dict_interface() -> None:
    user = User(identity="test-user", is_authenticated=True, permissions=["read"])
    assert user["identity"] == "test-user"
    assert "identity" in user
    assert "nonexistent" not in user
    keys = list(user)
    assert "identity" in keys
    assert "is_authenticated" in keys
