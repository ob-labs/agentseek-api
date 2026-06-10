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


def test_get_auth_backend_caches_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", None)
    monkeypatch.setattr(auth_middleware, "get_config_auth_settings", lambda: auth_middleware.ConfigAuthSettings())
    auth_middleware._backend = None
    backend1 = get_auth_backend()
    backend2 = get_auth_backend()
    assert backend1 is backend2


def test_get_auth_backend_defaults_to_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", None)
    monkeypatch.setattr(auth_middleware, "get_config_auth_settings", lambda: auth_middleware.ConfigAuthSettings())
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
async def test_langgraph_auth_backend_injects_all_supported_params(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    auth_file = tmp_path / "full_params_auth.py"
    auth_file.write_text(
        """
from langgraph_sdk import Auth

auth = Auth()

@auth.authenticate
async def get_current_user(request, headers: dict, method: str, path: str, query_params: dict, path_params: dict):
    api_key = headers.get(b"x-api-key", b"").decode()
    if not api_key:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Missing key")
    return {"identity": f"{method}:{path}:{api_key}"}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", f"{auth_file}:auth")
    auth_middleware._backend = None

    backend = get_auth_backend()
    assert isinstance(backend, LangGraphAuthBackend)

    request = Request({
        "type": "http",
        "method": "POST",
        "path": "/threads",
        "query_string": b"foo=bar",
        "headers": [(b"x-api-key", b"my-key")],
        "path_params": {"thread_id": "t-123"},
    })
    user = await backend.authenticate(request)
    assert user.identity == "POST:/threads:my-key"
    assert user.is_authenticated is True

    no_key_request = Request({
        "type": "http",
        "method": "GET",
        "path": "/threads",
        "query_string": b"",
        "headers": [],
        "path_params": {},
    })
    user = await backend.authenticate(no_key_request)
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


def test_get_studio_user_rejects_wrong_auth_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", "some_module:auth")
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", None)
    monkeypatch.setattr(settings, "STUDIO_AUTH_LOCAL_DEV", True)

    request = Request(
        {
            "type": "http",
            "headers": [(b"x-auth-scheme", b"bearer")],
            "client": ("127.0.0.1", 50000),
        }
    )

    assert get_studio_user(request) is None


def test_get_studio_user_rejects_when_auth_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", None)
    monkeypatch.setattr(settings, "STUDIO_AUTH_LOCAL_DEV", True)
    monkeypatch.setattr(auth_middleware, "get_config_auth_settings", lambda: auth_middleware.ConfigAuthSettings())

    request = Request(
        {
            "type": "http",
            "headers": [(b"x-auth-scheme", b"langsmith")],
            "client": ("127.0.0.1", 50000),
        }
    )

    assert get_studio_user(request) is None


def test_is_loopback_client_handles_no_client() -> None:
    request = Request({"type": "http", "headers": []})
    assert auth_middleware._is_loopback_client(request) is False


def test_is_loopback_client_handles_localhost_string() -> None:
    request = Request({"type": "http", "headers": [], "client": ("localhost", 50000)})
    assert auth_middleware._is_loopback_client(request) is True


def test_is_loopback_client_handles_invalid_host() -> None:
    request = Request({"type": "http", "headers": [], "client": ("not-an-ip", 50000)})
    assert auth_middleware._is_loopback_client(request) is False


def test_normalize_config_symbol_reference_without_colon() -> None:
    result = auth_middleware._normalize_config_symbol_reference("module_path_only", config_path=Path("/tmp/config.json"))
    assert result == "module_path_only"


def test_normalize_config_symbol_reference_module_style() -> None:
    result = auth_middleware._normalize_config_symbol_reference("mypackage.auth:handler", config_path=Path("/tmp/config.json"))
    assert result == "mypackage.auth:handler"


def test_apply_config_dependencies_skips_non_list(tmp_path: Path) -> None:
    auth_middleware._apply_config_dependencies({"dependencies": "not-a-list"}, config_path=tmp_path / "config.json")


def test_apply_config_dependencies_skips_non_string_entries(tmp_path: Path) -> None:
    auth_middleware._apply_config_dependencies({"dependencies": [123, None, "."]}, config_path=tmp_path / "config.json")


def test_apply_config_dependencies_handles_relative_path(tmp_path: Path) -> None:
    lib_dir = tmp_path / "libs"
    lib_dir.mkdir()
    auth_middleware._apply_config_dependencies({"dependencies": [str(lib_dir)]}, config_path=tmp_path / "config.json")
    import sys
    assert str(lib_dir.resolve()) in sys.path


def test_get_auth_backend_rejects_empty_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", "module:")
    auth_middleware._backend = None

    with pytest.raises(RuntimeError, match="Invalid AUTH_MODULE_PATH"):
        get_auth_backend()
