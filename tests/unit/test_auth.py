import types
import base64
import hashlib
import hmac
import json
import time
from pathlib import Path

import pytest
from fastapi import Request

from agentseek_api.core import auth_middleware
from agentseek_api.core.auth_middleware import (
    ApiKeyAuthBackend,
    JwtAuthBackend,
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
    monkeypatch.setattr(settings, "AUTH_TYPE", "noop")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
    auth_middleware._backend = None
    backend = get_auth_backend()
    assert isinstance(backend, NoopAuthBackend)


@pytest.mark.asyncio
async def test_api_key_auth_backend_maps_header_key_to_user() -> None:
    backend = ApiKeyAuthBackend("key-one=user-1,key-two=user-2")
    request = Request({"type": "http", "headers": [(b"x-api-key", b"key-two")]})

    user = await backend.authenticate(request)

    assert user.identity == "user-2"
    assert user.is_authenticated is True


@pytest.mark.asyncio
async def test_api_key_auth_backend_rejects_missing_or_unknown_key() -> None:
    backend = ApiKeyAuthBackend("known=user")

    missing = await backend.authenticate(Request({"type": "http", "headers": []}))
    unknown = await backend.authenticate(Request({"type": "http", "headers": [(b"x-api-key", b"unknown")]}))

    assert missing.identity == "anonymous"
    assert missing.is_authenticated is False
    assert unknown.identity == "anonymous"
    assert unknown.is_authenticated is False


def test_api_key_auth_backend_rejects_malformed_mapping() -> None:
    with pytest.raises(RuntimeError, match="AUTH_API_KEYS entries must use 'key=user_id' format"):
        ApiKeyAuthBackend("missing-user")


def _signed_hs256_jwt(payload: dict[str, object], secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}

    def encode(value: dict[str, object]) -> str:
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    signing_input = f"{encode(header)}.{encode(payload)}"
    signature = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
    return f"{signing_input}.{encoded_signature}"


@pytest.mark.asyncio
async def test_jwt_auth_backend_uses_sub_claim_identity() -> None:
    token = _signed_hs256_jwt({"sub": "jwt-user"}, "secret")
    backend = JwtAuthBackend(secret="secret", algorithm="HS256")
    request = Request({"type": "http", "headers": [(b"authorization", f"Bearer {token}".encode())]})

    user = await backend.authenticate(request)

    assert user.identity == "jwt-user"
    assert user.is_authenticated is True


@pytest.mark.asyncio
async def test_jwt_auth_backend_rejects_invalid_signature() -> None:
    token = _signed_hs256_jwt({"sub": "jwt-user"}, "wrong-secret")
    backend = JwtAuthBackend(secret="secret", algorithm="HS256")
    request = Request({"type": "http", "headers": [(b"authorization", f"Bearer {token}".encode())]})

    user = await backend.authenticate(request)

    assert user.identity == "anonymous"
    assert user.is_authenticated is False


@pytest.mark.asyncio
async def test_jwt_auth_backend_rejects_expired_token() -> None:
    token = _signed_hs256_jwt({"sub": "jwt-user", "exp": int(time.time()) - 60}, "secret")
    backend = JwtAuthBackend(secret="secret", algorithm="HS256")
    request = Request({"type": "http", "headers": [(b"authorization", f"Bearer {token}".encode())]})

    user = await backend.authenticate(request)

    assert user.identity == "anonymous"
    assert user.is_authenticated is False


@pytest.mark.asyncio
async def test_jwt_auth_backend_rejects_not_yet_valid_token() -> None:
    token = _signed_hs256_jwt({"sub": "jwt-user", "nbf": int(time.time()) + 60}, "secret")
    backend = JwtAuthBackend(secret="secret", algorithm="HS256")
    request = Request({"type": "http", "headers": [(b"authorization", f"Bearer {token}".encode())]})

    user = await backend.authenticate(request)

    assert user.identity == "anonymous"
    assert user.is_authenticated is False


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
    monkeypatch.setattr(settings, "AUTH_TYPE", "custom")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", f"{auth_file}:backend")
    auth_middleware._backend = None

    backend = get_auth_backend()
    user = await backend.authenticate(Request({"type": "http", "headers": []}))
    assert user.identity == "file_user"


@pytest.mark.asyncio
async def test_get_auth_backend_loads_custom_backend_from_json_config(
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
    "path": "./auth.py:auth",
    "openapi": {
      "securitySchemes": {
        "apiKeyAuth": {
          "type": "apiKey",
          "in": "header",
          "name": "X-API-Key"
        }
      },
      "security": [{ "apiKeyAuth": [] }]
    },
    "disable_studio_auth": false
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AUTH_TYPE", "noop")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))
    auth_middleware._backend = None

    backend = get_auth_backend()
    user = await backend.authenticate(Request({"type": "http", "headers": []}))

    assert user.identity == "config_user"


def test_get_auth_backend_custom_requires_module_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_TYPE", "custom")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
    auth_middleware._backend = None

    with pytest.raises(RuntimeError, match="AUTH_TYPE=custom requires AUTH_MODULE_PATH"):
        get_auth_backend()


def test_get_auth_backend_loads_api_key_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_TYPE", "api_key")
    monkeypatch.setattr(settings, "AUTH_API_KEYS", "key=user")
    auth_middleware._backend = None

    backend = get_auth_backend()

    assert isinstance(backend, ApiKeyAuthBackend)


def test_get_auth_backend_loads_jwt_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_TYPE", "jwt")
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", "secret")
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    auth_middleware._backend = None

    backend = get_auth_backend()

    assert isinstance(backend, JwtAuthBackend)


def test_get_auth_backend_jwt_requires_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_TYPE", "jwt")
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", None)
    auth_middleware._backend = None

    with pytest.raises(RuntimeError, match="AUTH_TYPE=jwt requires AUTH_JWT_SECRET"):
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


def test_get_studio_user_rejects_non_loopback_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AUTH_TYPE", "api_key")
    monkeypatch.setattr(settings, "AUTH_API_KEYS", "secret=user")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
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
    monkeypatch.setattr(settings, "AUTH_TYPE", "api_key")
    monkeypatch.setattr(settings, "AUTH_API_KEYS", "secret=user")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
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
    monkeypatch.setattr(settings, "AUTH_TYPE", "api_key")
    monkeypatch.setattr(settings, "AUTH_API_KEYS", "secret=user")
    monkeypatch.setattr(settings, "AUTH_MODULE_PATH", None)
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
