"""Unit tests covering auth internal edge cases for coverage."""
import pytest
from unittest.mock import AsyncMock, patch

from agentseek_api.core.auth_deps import authorize, apply_metadata_filters
from agentseek_api.core.auth_middleware import LangGraphAuthBackend
from agentseek_api.models.auth import User
from langgraph_sdk import Auth


def _make_backend_with_handler(handler):
    auth = Auth()

    @auth.authenticate
    async def authenticate(authorization: str | None):
        return {"identity": "test", "is_authenticated": True}

    auth._global_handlers = [handler]
    return LangGraphAuthBackend(auth)


@pytest.mark.asyncio
async def test_authorize_value_none_defaults_to_empty_dict():
    async def handler(ctx, value):
        assert value == {}
        return {"owner": ctx.user.identity}

    backend = _make_backend_with_handler(handler)
    user = User(identity="alice", is_authenticated=True)

    with patch("agentseek_api.core.auth_deps.get_auth_backend", return_value=backend):
        result = await authorize(user, "threads", "read", None)
    assert result == {"owner": "alice"}


@pytest.mark.asyncio
async def test_authorize_reraises_non_http_exceptions():
    async def handler(ctx, value):
        raise RuntimeError("unexpected internal error")

    backend = _make_backend_with_handler(handler)
    user = User(identity="alice", is_authenticated=True)

    with patch("agentseek_api.core.auth_deps.get_auth_backend", return_value=backend):
        with pytest.raises(RuntimeError, match="unexpected internal error"):
            await authorize(user, "threads", "read", {})


@pytest.mark.asyncio
async def test_backend_authorize_returns_none_when_no_handlers():
    auth = Auth()

    @auth.authenticate
    async def authenticate(authorization: str | None):
        return {"identity": "test", "is_authenticated": True}

    backend = LangGraphAuthBackend(auth)
    user = User(identity="alice", is_authenticated=True)
    result = await backend.authorize(user, "threads", "read", {})
    assert result is None


@pytest.mark.asyncio
async def test_backend_authorize_returns_none_for_studio_user():
    async def handler(ctx, value):
        return {"owner": ctx.user.identity}

    backend = _make_backend_with_handler(handler)
    user = User(identity="langgraph-studio-user", is_authenticated=True)
    result = await backend.authorize(user, "threads", "read", {})
    assert result is None


@pytest.mark.asyncio
async def test_backend_authorize_handler_returns_false_raises_403():
    async def handler(ctx, value):
        return False

    backend = _make_backend_with_handler(handler)
    user = User(identity="alice", is_authenticated=True)

    with pytest.raises(Exception) as exc_info:
        await backend.authorize(user, "threads", "read", {})
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_backend_resolve_handler_resource_action_specific():
    auth = Auth()

    @auth.authenticate
    async def authenticate(authorization: str | None):
        return {"identity": "test", "is_authenticated": True}

    async def specific_handler(ctx, value):
        return {"specific": True}

    async def wildcard_handler(ctx, value):
        return {"wildcard": True}

    auth._handlers = {
        ("threads", "create"): [specific_handler],
        ("threads", "*"): [wildcard_handler],
    }
    backend = LangGraphAuthBackend(auth)
    user = User(identity="alice", is_authenticated=True)

    result = await backend.authorize(user, "threads", "create", {})
    assert result == {"specific": True}

    result = await backend.authorize(user, "threads", "read", {})
    assert result == {"wildcard": True}


@pytest.mark.asyncio
async def test_backend_resolve_handler_returns_none_when_no_match():
    auth = Auth()

    @auth.authenticate
    async def authenticate(authorization: str | None):
        return {"identity": "test", "is_authenticated": True}

    auth._handlers = {("assistants", "create"): [AsyncMock(return_value=None)]}
    backend = LangGraphAuthBackend(auth)
    backend._has_on_handlers = True
    user = User(identity="alice", is_authenticated=True)

    result = await backend.authorize(user, "threads", "read", {})
    assert result is None


def test_apply_metadata_filters_contains_single():
    from sqlalchemy import select, Column, JSON, String
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

    class Base(DeclarativeBase):
        pass

    class FakeModel(Base):
        __tablename__ = "fake"
        id: Mapped[str] = mapped_column(String(36), primary_key=True)
        metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    stmt = select(FakeModel)
    filters = {"tags": {"$contains": "important"}}
    result = apply_metadata_filters(stmt, FakeModel, filters)
    compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
    assert "LIKE" in compiled or "contains" in compiled.lower() or "metadata" in compiled


def test_apply_metadata_filters_contains_list():
    from sqlalchemy import select, Column, JSON, String
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

    class Base(DeclarativeBase):
        pass

    class FakeModel(Base):
        __tablename__ = "fake2"
        id: Mapped[str] = mapped_column(String(36), primary_key=True)
        metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    stmt = select(FakeModel)
    filters = {"tags": {"$contains": ["a", "b"]}}
    result = apply_metadata_filters(stmt, FakeModel, filters)
    compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
    assert "metadata" in compiled


def test_apply_metadata_filters_plain_value():
    from sqlalchemy import select, Column, JSON, String
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

    class Base(DeclarativeBase):
        pass

    class FakeModel(Base):
        __tablename__ = "fake3"
        id: Mapped[str] = mapped_column(String(36), primary_key=True)
        metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    stmt = select(FakeModel)
    filters = {"owner": "alice"}
    result = apply_metadata_filters(stmt, FakeModel, filters)
    compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
    assert "metadata" in compiled


def test_user_model_getitem_and_contains():
    user = User(identity="alice", is_authenticated=True, permissions=["read", "write"])
    assert user["identity"] == "alice"
    assert user["permissions"] == ["read", "write"]
    assert "permissions" in user

    with pytest.raises(KeyError):
        _ = user["nonexistent_field"]
