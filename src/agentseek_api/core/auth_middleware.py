import base64
import hashlib
import hmac
import json
import sys
from dataclasses import dataclass
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any
from typing import Protocol

from fastapi import Request

from agentseek_api.models.auth import User
from agentseek_api.settings import settings


class AuthBackend(Protocol):
    async def authenticate(self, request: Request) -> User: ...


@dataclass(frozen=True)
class ConfigAuthSettings:
    path: str | None = None
    openapi: dict[str, Any] | None = None
    disable_studio_auth: bool | None = None


class NoopAuthBackend:
    async def authenticate(self, _request: Request) -> User:
        return User(identity="default_user", is_authenticated=True)


class ApiKeyAuthBackend:
    def __init__(self, api_keys: str) -> None:
        self._users_by_key = _parse_api_key_mapping(api_keys)

    async def authenticate(self, request: Request) -> User:
        api_key = request.headers.get("x-api-key")
        if not api_key:
            return User(identity="anonymous", is_authenticated=False)
        identity = self._users_by_key.get(api_key)
        if identity is None:
            return User(identity="anonymous", is_authenticated=False)
        return User(identity=identity, is_authenticated=True)


class JwtAuthBackend:
    def __init__(self, *, secret: str, algorithm: str = "HS256") -> None:
        if algorithm != "HS256":
            raise RuntimeError("Only AUTH_JWT_ALGORITHM=HS256 is supported.")
        self._secret = secret
        self._algorithm = algorithm

    async def authenticate(self, request: Request) -> User:
        authorization = request.headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return User(identity="anonymous", is_authenticated=False)

        payload = _decode_hs256_jwt(token, secret=self._secret, algorithm=self._algorithm)
        subject = payload.get("sub") if payload is not None else None
        if not isinstance(subject, str) or not subject:
            return User(identity="anonymous", is_authenticated=False)
        return User(identity=subject, is_authenticated=True)


def _parse_api_key_mapping(raw_value: str) -> dict[str, str]:
    users_by_key: dict[str, str] = {}
    for raw_entry in raw_value.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise RuntimeError("AUTH_API_KEYS entries must use 'key=user_id' format.")
        key, identity = (part.strip() for part in entry.split("=", maxsplit=1))
        if not key or not identity:
            raise RuntimeError("AUTH_API_KEYS entries must use 'key=user_id' format.")
        users_by_key[key] = identity
    if not users_by_key:
        raise RuntimeError("AUTH_TYPE=api_key requires AUTH_API_KEYS to contain at least one key=user_id entry.")
    return users_by_key


def _decode_urlsafe_json(segment: str) -> dict[str, Any] | None:
    try:
        padded = segment + ("=" * (-len(segment) % 4))
        raw = base64.urlsafe_b64decode(padded.encode())
        payload = json.loads(raw)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _decode_hs256_jwt(token: str, *, secret: str, algorithm: str) -> dict[str, Any] | None:
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
    except ValueError:
        return None
    header = _decode_urlsafe_json(header_segment)
    if header is None or header.get("alg") != algorithm:
        return None

    signing_input = f"{header_segment}.{payload_segment}".encode()
    expected_signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    expected_segment = base64.urlsafe_b64encode(expected_signature).rstrip(b"=").decode()
    if not hmac.compare_digest(signature_segment, expected_segment):
        return None
    return _decode_urlsafe_json(payload_segment)


def _load_python_file_backend(module_ref: str) -> object:
    file_path = Path(module_ref).expanduser().resolve()
    module_name = f"agentseek_auth_{abs(hash(file_path))}"
    spec = spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load AUTH_MODULE_PATH module file '{file_path}'.")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _active_config_path() -> Path | None:
    if settings.AGENTSEEK_GRAPHS:
        path = Path(settings.AGENTSEEK_GRAPHS).expanduser().resolve()
        if path.exists():
            return path
    for candidate in ("agentseek.json", "langgraph.json"):
        path = Path(candidate).resolve()
        if path.exists():
            return path
    return None


def _apply_config_dependencies(payload: dict[str, Any], *, config_path: Path) -> None:
    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, list):
        return
    for dependency in dependencies:
        if not isinstance(dependency, str):
            continue
        if dependency == ".":
            root = config_path.parent.resolve()
        else:
            candidate = Path(dependency).expanduser()
            root = candidate.resolve() if candidate.is_absolute() else (config_path.parent / candidate).resolve()
        if root.exists():
            root_text = str(root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)


def _normalize_config_symbol_reference(reference: str, *, config_path: Path) -> str:
    if ":" not in reference:
        return reference
    module_ref, symbol = reference.rsplit(":", maxsplit=1)
    if module_ref.endswith(".py") or module_ref.startswith(".") or "/" in module_ref or "\\" in module_ref:
        module_path = Path(module_ref).expanduser()
        if not module_path.is_absolute():
            module_path = config_path.parent / module_path
        return f"{module_path.resolve()}:{symbol}"
    return reference


def get_config_auth_settings() -> ConfigAuthSettings:
    config_path = _active_config_path()
    if config_path is None:
        return ConfigAuthSettings()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ConfigAuthSettings()
    if not isinstance(payload, dict):
        return ConfigAuthSettings()
    raw_auth = payload.get("auth")
    if not isinstance(raw_auth, dict):
        return ConfigAuthSettings()
    _apply_config_dependencies(payload, config_path=config_path)

    raw_path = raw_auth.get("path")
    auth_path = _normalize_config_symbol_reference(raw_path, config_path=config_path) if isinstance(raw_path, str) else None
    raw_openapi = raw_auth.get("openapi")
    disable_studio_auth = raw_auth.get("disable_studio_auth")
    return ConfigAuthSettings(
        path=auth_path,
        openapi=raw_openapi if isinstance(raw_openapi, dict) else None,
        disable_studio_auth=disable_studio_auth if isinstance(disable_studio_auth, bool) else None,
    )


def get_config_auth_openapi() -> dict[str, Any] | None:
    return get_config_auth_settings().openapi


def _load_custom_backend(auth_module_path: str | None = None) -> AuthBackend | None:
    auth_module_path = auth_module_path or settings.AUTH_MODULE_PATH
    if not auth_module_path:
        return None
    if ":" not in auth_module_path:
        raise RuntimeError(
            f"Invalid AUTH_MODULE_PATH='{auth_module_path}'. Expected format 'module.path:symbol'."
        )

    module_name, symbol = auth_module_path.rsplit(":", maxsplit=1)
    if not module_name or not symbol:
        raise RuntimeError(
            f"Invalid AUTH_MODULE_PATH='{auth_module_path}'. Expected format 'module.path:symbol'."
        )

    try:
        if module_name.endswith(".py") or module_name.startswith(".") or "/" in module_name or "\\" in module_name:
            module = _load_python_file_backend(module_name)
        else:
            module = import_module(module_name)
        obj = getattr(module, symbol)
        return obj() if callable(obj) else obj
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not load AUTH_MODULE_PATH='{auth_module_path}': {exc}"
        ) from exc


_backend: AuthBackend | None = None


def get_auth_backend() -> AuthBackend:
    global _backend
    if _backend is not None:
        return _backend

    config_auth = get_config_auth_settings()
    auth_type = settings.AUTH_TYPE.strip().lower()
    if auth_type == "noop":
        if config_auth.path:
            custom_backend = _load_custom_backend(config_auth.path)
            if custom_backend is not None:
                _backend = custom_backend
                return _backend
        _backend = NoopAuthBackend()
        return _backend
    if auth_type == "custom":
        custom_backend = _load_custom_backend(settings.AUTH_MODULE_PATH or config_auth.path)
        if custom_backend is None:
            raise RuntimeError("AUTH_TYPE=custom requires AUTH_MODULE_PATH to be configured.")
        _backend = custom_backend
        return _backend
    if auth_type == "api_key":
        if not settings.AUTH_API_KEYS:
            raise RuntimeError("AUTH_TYPE=api_key requires AUTH_API_KEYS to be configured.")
        _backend = ApiKeyAuthBackend(settings.AUTH_API_KEYS)
        return _backend
    if auth_type == "jwt":
        if not settings.AUTH_JWT_SECRET:
            raise RuntimeError("AUTH_TYPE=jwt requires AUTH_JWT_SECRET to be configured.")
        _backend = JwtAuthBackend(secret=settings.AUTH_JWT_SECRET, algorithm=settings.AUTH_JWT_ALGORITHM)
        return _backend
    raise ValueError(f"Unsupported AUTH_TYPE: {settings.AUTH_TYPE}")

    return _backend
