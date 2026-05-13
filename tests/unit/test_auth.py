import types

import pytest
from fastapi import Request

from agentseek_api.core import auth_middleware
from agentseek_api.core.auth_middleware import NoopAuthBackend, get_auth_backend
from agentseek_api.models.auth import User
from agentseek_api.settings import settings


@pytest.mark.asyncio
async def test_noop_auth_backend_returns_default_user() -> None:
    backend = NoopAuthBackend()
    request = Request({"type": "http", "headers": []})
    user = await backend.authenticate(request)
    assert user.identity == "default_user"
    assert user.is_authenticated is True


def test_get_auth_backend_defaults_to_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_TYPE", "noop")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
    auth_middleware._backend = None
    backend = get_auth_backend()
    assert isinstance(backend, NoopAuthBackend)


@pytest.mark.asyncio
async def test_get_auth_backend_loads_custom_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class CustomBackend:
        async def authenticate(self, _request: Request) -> User:
            return User(identity="custom_user", is_authenticated=True)

    module = types.ModuleType("fake_auth_module")
    module.backend = CustomBackend
    monkeypatch.setattr(auth_middleware, "import_module", lambda _name: module)
    monkeypatch.setattr(settings, "AUTH_TYPE", "custom")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", "fake_auth_module:backend")
    auth_middleware._backend = None

    backend = get_auth_backend()
    user = await backend.authenticate(Request({"type": "http", "headers": []}))
    assert user.identity == "custom_user"


def test_get_auth_backend_custom_requires_module_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_TYPE", "custom")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
    auth_middleware._backend = None

    with pytest.raises(RuntimeError, match="AUTH_TYPE=custom requires AUTH_MODULE_PATH"):
        get_auth_backend()


def test_get_auth_backend_rejects_unknown_auth_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_TYPE", "invalid")
    auth_middleware._backend = None

    with pytest.raises(ValueError, match="Unsupported AUTH_TYPE"):
        get_auth_backend()


def test_get_auth_backend_rejects_malformed_module_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_TYPE", "custom")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", "missing_separator")
    auth_middleware._backend = None

    with pytest.raises(RuntimeError, match="Invalid AUTH_MODULE_PATH"):
        get_auth_backend()


def test_get_auth_backend_wraps_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_TYPE", "custom")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", "missing.module:backend")
    auth_middleware._backend = None
    monkeypatch.setattr(
        auth_middleware,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'missing.module'")),
    )

    with pytest.raises(RuntimeError, match="Could not load AUTH_MODULE_PATH='missing.module:backend'"):
        get_auth_backend()


def test_get_auth_backend_wraps_missing_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("fake_auth_module")
    monkeypatch.setattr(settings, "AUTH_TYPE", "custom")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", "fake_auth_module:backend")
    auth_middleware._backend = None
    monkeypatch.setattr(auth_middleware, "import_module", lambda _name: module)

    with pytest.raises(RuntimeError, match="Could not load AUTH_MODULE_PATH='fake_auth_module:backend'"):
        get_auth_backend()
