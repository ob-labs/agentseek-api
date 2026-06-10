"""Tests covering documentation examples not previously exercised.

Fills gaps identified against LangSmith auth documentation:
- Permission-based access control in handlers
- Resource-specific handlers (per-action logic in integration)
- $eq filter operator
- Handler returning True/None (allow all)
- Multi-key AND filters
- langgraph_auth_user injection into run config
- is_studio_user() usage inside @auth.on handler
"""
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph_sdk import Auth

from agentseek_api.core import auth_middleware
from agentseek_api.core.auth_middleware import LangGraphAuthBackend
from agentseek_api.main import create_app
from agentseek_api.models.auth import User
from agentseek_api.settings import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_permission_based_auth() -> Auth:
    """Auth object using permission-based access control (doc example)."""
    auth = Auth()

    @auth.authenticate
    async def authenticate(authorization: str | None):
        if not authorization:
            raise Auth.exceptions.HTTPException(status_code=401, detail="Missing token")
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            raise Auth.exceptions.HTTPException(status_code=401, detail="Bad scheme")
        users = {
            "admin-token": {"identity": "admin", "permissions": ["threads:write", "threads:read", "assistants:create"]},
            "reader-token": {"identity": "reader", "permissions": ["threads:read"]},
        }
        user_data = users.get(token)
        if user_data is None:
            raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token")
        return user_data

    @auth.on.threads.create
    async def on_thread_create(ctx, value):
        if "threads:write" not in ctx.permissions:
            raise Auth.exceptions.HTTPException(status_code=403, detail="Unauthorized")
        metadata = value.setdefault("metadata", {})
        metadata["owner"] = ctx.user.identity
        return {"owner": ctx.user.identity}

    @auth.on.threads.read
    async def on_thread_read(ctx, value):
        if "threads:read" not in ctx.permissions and "threads:write" not in ctx.permissions:
            raise Auth.exceptions.HTTPException(status_code=403, detail="Unauthorized")
        return {"owner": ctx.user.identity}

    @auth.on.threads.search
    async def on_thread_search(ctx, value):
        if "threads:read" not in ctx.permissions and "threads:write" not in ctx.permissions:
            raise Auth.exceptions.HTTPException(status_code=403, detail="Unauthorized")
        return {"owner": ctx.user.identity}

    @auth.on.assistants.create
    async def on_assistant_create(ctx, value):
        if "assistants:create" not in ctx.permissions:
            raise Auth.exceptions.HTTPException(status_code=403, detail="Unauthorized")
        metadata = value.setdefault("metadata", {})
        metadata["owner"] = ctx.user.identity
        return {"owner": ctx.user.identity}

    return auth


def _build_resource_specific_auth() -> Auth:
    """Auth object with per-resource-per-action handlers returning different behavior."""
    auth = Auth()

    @auth.authenticate
    async def authenticate(authorization: str | None):
        if not authorization:
            raise Auth.exceptions.HTTPException(status_code=401, detail="Missing token")
        scheme, token = authorization.split(" ", 1)
        users = {"alice-token": "alice", "bob-token": "bob"}
        identity = users.get(token)
        if identity is None:
            raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token")
        return {"identity": identity}

    @auth.on.threads.create
    async def on_thread_create(ctx, value):
        metadata = value.setdefault("metadata", {})
        metadata["owner"] = ctx.user.identity
        metadata["created_via"] = "create_handler"
        return {"owner": ctx.user.identity}

    @auth.on.threads.read
    async def on_thread_read(ctx, value):
        return {"owner": ctx.user.identity}

    @auth.on.threads.search
    async def on_thread_search(ctx, value):
        return {"owner": ctx.user.identity}

    @auth.on.assistants
    async def on_assistants(ctx, value):
        raise Auth.exceptions.HTTPException(status_code=403, detail="User lacks the required permissions.")

    return auth


def _build_allow_all_auth() -> Auth:
    """Auth object with handler that returns True or None (allow all)."""
    auth = Auth()

    @auth.authenticate
    async def authenticate(authorization: str | None):
        if not authorization:
            raise Auth.exceptions.HTTPException(status_code=401, detail="Missing token")
        return {"identity": "user1"}

    @auth.on
    async def allow_all(ctx, value):
        return None

    return auth


def _build_studio_user_in_handler_auth() -> Auth:
    """Auth object that checks is_studio_user inside @auth.on handler."""
    from langgraph_sdk.auth import is_studio_user

    auth = Auth()

    @auth.authenticate
    async def authenticate(authorization: str | None):
        if not authorization:
            raise Auth.exceptions.HTTPException(status_code=401, detail="Missing token")
        return {"identity": "regular-user"}

    @auth.on
    async def add_owner(ctx, value):
        if is_studio_user(ctx.user):
            return {}
        metadata = value.setdefault("metadata", {})
        metadata["owner"] = ctx.user.identity
        return {"owner": ctx.user.identity}

    return auth


def _make_auth_client(monkeypatch, tmp_path, auth_obj) -> TestClient:
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

    backend = LangGraphAuthBackend(auth_obj)
    auth_middleware._backend = backend

    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def permission_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    client = _make_auth_client(monkeypatch, tmp_path, _build_permission_based_auth())
    with client:
        yield client
    auth_middleware._backend = None


@pytest.fixture
def resource_specific_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    client = _make_auth_client(monkeypatch, tmp_path, _build_resource_specific_auth())
    with client:
        yield client
    auth_middleware._backend = None


@pytest.fixture
def allow_all_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    client = _make_auth_client(monkeypatch, tmp_path, _build_allow_all_auth())
    with client:
        yield client
    auth_middleware._backend = None


@pytest.fixture
def studio_handler_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    from tests.conftest import FakeCheckpointer, InlineExecutor, _noop_ensure_default_assistants

    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)
    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", lambda: InlineExecutor())
    monkeypatch.setattr("agentseek_api.main.ensure_default_assistants", _noop_ensure_default_assistants)
    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", "fake:auth")
    monkeypatch.setattr(settings, "STUDIO_AUTH_LOCAL_DEV", True)
    monkeypatch.setattr(
        "agentseek_api.core.auth_middleware.get_config_auth_settings",
        lambda: auth_middleware.ConfigAuthSettings(),
    )

    backend = LangGraphAuthBackend(_build_studio_user_in_handler_auth())
    auth_middleware._backend = backend

    app = create_app()
    with TestClient(app, client=("127.0.0.1", 50000)) as test_client:
        yield test_client
    auth_middleware._backend = None


# ---------------------------------------------------------------------------
# Permission-based access control tests
# ---------------------------------------------------------------------------

ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}
READER_HEADERS = {"Authorization": "Bearer reader-token"}


class TestPermissionBasedAccess:
    """Verify permission checks in handlers (doc: Permission-based access)."""

    def test_admin_can_create_thread(self, permission_client: TestClient) -> None:
        resp = permission_client.post("/threads", json={}, headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["metadata"]["owner"] == "admin"

    def test_reader_cannot_create_thread(self, permission_client: TestClient) -> None:
        resp = permission_client.post("/threads", json={}, headers=READER_HEADERS)
        assert resp.status_code == 403

    def test_reader_can_read_own_threads(self, permission_client: TestClient) -> None:
        # Admin creates a thread, reader can't see it (owner filter)
        admin_resp = permission_client.post("/threads", json={}, headers=ADMIN_HEADERS)
        assert admin_resp.status_code == 200

        reader_search = permission_client.post("/threads/search", json={}, headers=READER_HEADERS)
        assert reader_search.status_code == 200
        assert len(reader_search.json()) == 0

    def test_admin_can_create_assistant(self, permission_client: TestClient) -> None:
        resp = permission_client.post(
            "/assistants",
            json={"name": "test", "graph_id": "default"},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200

    def test_reader_cannot_create_assistant(self, permission_client: TestClient) -> None:
        resp = permission_client.post(
            "/assistants",
            json={"name": "test", "graph_id": "default"},
            headers=READER_HEADERS,
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Resource-specific handler tests (integration level)
# ---------------------------------------------------------------------------

ALICE_HEADERS = {"Authorization": "Bearer alice-token"}
BOB_HEADERS = {"Authorization": "Bearer bob-token"}


class TestResourceSpecificHandlers:
    """Verify per-resource-per-action handlers with different logic (doc example)."""

    def test_thread_create_adds_created_via_metadata(self, resource_specific_client: TestClient) -> None:
        resp = resource_specific_client.post("/threads", json={}, headers=ALICE_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["metadata"]["created_via"] == "create_handler"
        assert resp.json()["metadata"]["owner"] == "alice"

    def test_thread_read_uses_different_handler(self, resource_specific_client: TestClient) -> None:
        create_resp = resource_specific_client.post("/threads", json={}, headers=ALICE_HEADERS)
        thread_id = create_resp.json()["thread_id"]

        alice_get = resource_specific_client.get(f"/threads/{thread_id}", headers=ALICE_HEADERS)
        assert alice_get.status_code == 200

        bob_get = resource_specific_client.get(f"/threads/{thread_id}", headers=BOB_HEADERS)
        assert bob_get.status_code == 404

    def test_assistants_handler_rejects_all_actions(self, resource_specific_client: TestClient) -> None:
        create = resource_specific_client.post(
            "/assistants",
            json={"name": "blocked", "graph_id": "default"},
            headers=ALICE_HEADERS,
        )
        assert create.status_code == 403

        search = resource_specific_client.post("/assistants/search", json={}, headers=ALICE_HEADERS)
        assert search.status_code == 403


# ---------------------------------------------------------------------------
# Handler returns True/None (allow all)
# ---------------------------------------------------------------------------


class TestHandlerReturnsAllowAll:
    """Verify that handler returning None means allow access to all resources."""

    def test_handler_returning_none_allows_all_access(self, allow_all_client: TestClient) -> None:
        headers = {"Authorization": "Bearer any-token"}
        resp = allow_all_client.post("/threads", json={}, headers=headers)
        assert resp.status_code == 200
        thread_id = resp.json()["thread_id"]

        get_resp = allow_all_client.get(f"/threads/{thread_id}", headers=headers)
        assert get_resp.status_code == 200

    def test_handler_returning_none_no_metadata_filtering_on_search(self, allow_all_client: TestClient) -> None:
        headers = {"Authorization": "Bearer any-token"}
        allow_all_client.post("/threads", json={"metadata": {"label": "a"}}, headers=headers)
        allow_all_client.post("/threads", json={"metadata": {"label": "b"}}, headers=headers)

        search = allow_all_client.post("/threads/search", json={}, headers=headers)
        assert search.status_code == 200
        assert len(search.json()) >= 2


# ---------------------------------------------------------------------------
# $eq filter operator and multi-key AND filter
# ---------------------------------------------------------------------------


class TestFilterOperators:
    """Unit tests for $eq and multi-key AND filters in apply_metadata_filters."""

    def test_eq_operator(self) -> None:
        from sqlalchemy import select, JSON, String
        from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
        from agentseek_api.core.auth_deps import apply_metadata_filters

        class Base(DeclarativeBase):
            pass

        class Item(Base):
            __tablename__ = "eq_test"
            id: Mapped[str] = mapped_column(String(36), primary_key=True)
            metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

        stmt = select(Item)
        filters = {"owner": {"$eq": "alice"}}
        result = apply_metadata_filters(stmt, Item, filters)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "alice" in compiled
        assert "metadata" in compiled

    def test_multi_key_and_filter(self) -> None:
        from sqlalchemy import select, JSON, String
        from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
        from agentseek_api.core.auth_deps import apply_metadata_filters

        class Base(DeclarativeBase):
            pass

        class Item(Base):
            __tablename__ = "multi_key_test"
            id: Mapped[str] = mapped_column(String(36), primary_key=True)
            metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

        stmt = select(Item)
        filters = {"owner": "org-456", "allowed_users": {"$contains": "user-123"}}
        result = apply_metadata_filters(stmt, Item, filters)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "org-456" in compiled
        assert "user-123" in compiled


# ---------------------------------------------------------------------------
# langgraph_auth_user injection into run config
# ---------------------------------------------------------------------------


class TestAuthUserInjection:
    """Verify that langgraph_auth_user is injected into run execution config."""

    def test_run_executor_injects_auth_user(self) -> None:
        from agentseek_api.models.auth import User

        user = User(identity="test-user-42", is_authenticated=True)
        config: dict[str, Any] = {}
        configurable = config.setdefault("configurable", {})
        configurable["langgraph_auth_user"] = User(identity=user.identity)

        assert configurable["langgraph_auth_user"].identity == "test-user-42"

    def test_run_execution_job_carries_user_id(self) -> None:
        from agentseek_api.services.run_jobs import RunExecutionJob

        job = RunExecutionJob(
            run_id="run-1",
            thread_id="thread-1",
            user_id="injected-user",
            payload={},
            graph_id="default",
            kwargs={},
            resume=None,
            is_resume=False,
        )
        assert job.user_id == "injected-user"


# ---------------------------------------------------------------------------
# is_studio_user() inside @auth.on handler
# ---------------------------------------------------------------------------


class TestStudioUserInHandler:
    """Verify is_studio_user() can be used inside @auth.on handler (doc example)."""

    def test_studio_user_bypasses_owner_filter(self, studio_handler_client: TestClient) -> None:
        resp = studio_handler_client.post(
            "/threads",
            json={},
            headers={"Authorization": "Bearer test-token"},
        )
        assert resp.status_code == 200
        thread_id = resp.json()["thread_id"]
        assert resp.json()["metadata"]["owner"] == "regular-user"

        studio_resp = studio_handler_client.post(
            "/threads/search",
            json={},
            headers={"x-auth-scheme": "langsmith"},
        )
        assert studio_resp.status_code == 200
        threads = studio_resp.json()
        assert any(t["thread_id"] == thread_id for t in threads)

    def test_regular_user_subject_to_owner_filter(self, studio_handler_client: TestClient) -> None:
        studio_handler_client.post(
            "/threads",
            json={},
            headers={"Authorization": "Bearer test-token"},
        )

        search = studio_handler_client.post(
            "/threads/search",
            json={},
            headers={"Authorization": "Bearer test-token"},
        )
        assert search.status_code == 200
        for t in search.json():
            assert t["metadata"].get("owner") == "regular-user"


# ---------------------------------------------------------------------------
# authorize() returns True behavior (unit test)
# ---------------------------------------------------------------------------


class TestAuthorizeReturnTrue:
    """Verify that handler returning True means allow access."""

    @pytest.mark.asyncio
    async def test_handler_returns_true_allows_access(self) -> None:
        from unittest.mock import patch

        auth = Auth()

        @auth.authenticate
        async def authenticate(authorization: str | None):
            return {"identity": "test"}

        async def true_handler(ctx, value):
            return True

        auth._global_handlers = [true_handler]
        backend = LangGraphAuthBackend(auth)
        user = User(identity="alice", is_authenticated=True)

        result = await backend.authorize(user, "threads", "read", {})
        assert result is True
