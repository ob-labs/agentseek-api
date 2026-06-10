import types
from pathlib import Path

import pytest
from fastapi import Request

from agentseek_api.core import auth_middleware
from agentseek_api.core.auth_middleware import (
    LangGraphAuthBackend,
    NoopAuthBackend,
    get_auth_backend,
    get_studio_user,
)
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
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", None)
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
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", "fake_auth_module:backend")
    auth_middleware._backend = None

    backend = get_auth_backend()
    user = await backend.authenticate(Request({"type": "http", "headers": []}))
    assert user.identity == "custom_user"


@pytest.mark.asyncio
async def test_get_auth_backend_loads_custom_backend_from_python_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    auth_file = tmp_path / "custom_auth.py"
    auth_file.write_text(
        """
from dataclasses import dataclass

from agentseek_api.models.auth import User

@dataclass
class IdentityConfig:
    identity: str


DEFAULT_IDENTITY = IdentityConfig(identity="file_user")

class CustomBackend:
    async def authenticate(self, _request):
        return User(identity=DEFAULT_IDENTITY.identity, is_authenticated=True)

backend = CustomBackend
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", f"{auth_file}:backend")
    auth_middleware._backend = None

    backend = get_auth_backend()
    user = await backend.authenticate(Request({"type": "http", "headers": []}))
    assert user.identity == "file_user"


@pytest.mark.asyncio
async def test_get_auth_backend_loads_langgraph_sdk_auth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    auth_file = tmp_path / "sdk_auth.py"
    auth_file.write_text(
        """
from langgraph_sdk import Auth

auth = Auth()

@auth.authenticate
async def get_current_user(authorization: str | None):
    if not authorization:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Missing token")
    return {"identity": "sdk-user"}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", f"{auth_file}:auth")
    auth_middleware._backend = None

    backend = get_auth_backend()
    assert isinstance(backend, LangGraphAuthBackend)

    authed_request = Request({"type": "http", "headers": [(b"authorization", b"Bearer test-token")]})
    user = await backend.authenticate(authed_request)
    assert user.identity == "sdk-user"
    assert user.is_authenticated is True

    unauthed_request = Request({"type": "http", "headers": []})
    user = await backend.authenticate(unauthed_request)
    assert user.identity == "anonymous"
    assert user.is_authenticated is False


@pytest.mark.asyncio
async def test_get_auth_backend_loads_from_json_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "identity.py").write_text('IDENTITY = "config_user"\n', encoding="utf-8")
    auth_file = tmp_path / "auth.py"
    auth_file.write_text(
        """
from identity import IDENTITY

from agentseek_api.models.auth import User


class ConfigAuthBackend:
    async def authenticate(self, _request):
        return User(identity=IDENTITY, is_authenticated=True)


auth = ConfigAuthBackend
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        """
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "auth": {
    "path": "./auth.py:auth"
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))
    auth_middleware._backend = None

    backend = get_auth_backend()
    user = await backend.authenticate(Request({"type": "http", "headers": []}))

    assert user.identity == "config_user"


def test_get_auth_backend_rejects_malformed_module_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", "missing_separator")
    auth_middleware._backend = None

    with pytest.raises(RuntimeError, match="Invalid AUTH_MODULE_PATH"):
        get_auth_backend()


def test_get_auth_backend_wraps_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", "fake_auth_module:backend")
    auth_middleware._backend = None
    monkeypatch.setattr(auth_middleware, "import_module", lambda _name: module)

    with pytest.raises(RuntimeError, match="Could not load AUTH_MODULE_PATH='fake_auth_module:backend'"):
        get_auth_backend()


def test_get_studio_user_rejects_non_loopback_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", "some_module:auth")
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", None)
    monkeypatch.setattr(settings, "STUDIO_AUTH_LOCAL_DEV", True)

    request = Request(
        {
            "type": "http",
            "headers": [(b"x-auth-scheme", b"langsmith")],
            "client": ("203.0.113.10", 50000),
        }
    )

    assert get_studio_user(request) is None


def test_get_studio_user_rejects_loopback_requests_outside_local_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", "some_module:auth")
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", None)
    monkeypatch.setattr(settings, "STUDIO_AUTH_LOCAL_DEV", False)

    request = Request(
        {
            "type": "http",
            "headers": [(b"x-auth-scheme", b"langsmith")],
            "client": ("127.0.0.1", 50000),
        }
    )

    assert get_studio_user(request) is None


def test_get_studio_user_accepts_loopback_requests_in_local_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", "some_module:auth")
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", None)
    monkeypatch.setattr(settings, "STUDIO_AUTH_LOCAL_DEV", True)

    request = Request(
        {
            "type": "http",
            "headers": [(b"x-auth-scheme", b"langsmith")],
            "client": ("127.0.0.1", 50000),
        }
    )

    user = get_studio_user(request)

    assert user is not None
    assert user.identity == "langgraph-studio-user"
    assert user.is_authenticated is True
