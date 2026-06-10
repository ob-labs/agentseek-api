import inspect
import ipaddress
import sys
from dataclasses import dataclass
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any
from typing import Protocol

from fastapi import Request

from agentseek_api.core.config_file import active_config_path, get_active_config_payload
from agentseek_api.models.auth import User
from agentseek_api.settings import settings

STUDIO_AUTH_SCHEME = "langsmith"
STUDIO_USER_ID = "langgraph-studio-user"


class AuthBackend(Protocol):
    async def authenticate(self, request: Request) -> User: ...


@dataclass(frozen=True)
class ConfigAuthSettings:
    path: str | None = None
    openapi: dict[str, Any] | None = None
    disable_studio_auth: bool | None = None


class LangGraphAuthBackend:
    """Adapter that wraps a langgraph_sdk.Auth object into an AuthBackend."""

    def __init__(self, auth_obj) -> None:
        self._auth = auth_obj

    async def authenticate(self, request: Request) -> User:
        handler = self._auth._authenticate_handler
        params = inspect.signature(handler).parameters

        kwargs: dict[str, Any] = {}
        if "authorization" in params:
            kwargs["authorization"] = request.headers.get("authorization")
        if "headers" in params:
            kwargs["headers"] = {k.encode(): v.encode() for k, v in request.headers.items()}
        if "request" in params:
            kwargs["request"] = request
        if "method" in params:
            kwargs["method"] = request.method
        if "path" in params:
            kwargs["path"] = request.url.path
        if "path_params" in params:
            kwargs["path_params"] = request.path_params
        if "query_params" in params:
            kwargs["query_params"] = dict(request.query_params)

        try:
            result = await handler(**kwargs)
        except Exception:
            return User(identity="anonymous", is_authenticated=False)

        identity = result.get("identity", "anonymous") if isinstance(result, dict) else "anonymous"
        return User(identity=identity, is_authenticated=True)


class NoopAuthBackend:
    async def authenticate(self, _request: Request) -> User:
        return User(identity="default_user", is_authenticated=True)


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
    config_path = active_config_path()
    payload = get_active_config_payload()
    if config_path is None or payload is None:
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


def _auth_is_configured(config_auth: ConfigAuthSettings) -> bool:
    return bool(settings.AUTH_MODULE_PATH or config_auth.path)


def _is_loopback_client(request: Request) -> bool:
    client = request.client
    if client is None:
        return False
    host = client.host
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def get_studio_user(request: Request) -> User | None:
    config_auth = get_config_auth_settings()
    if config_auth.disable_studio_auth is True:
        return None
    if not settings.STUDIO_AUTH_LOCAL_DEV:
        return None
    auth_scheme = request.headers.get("x-auth-scheme", "")
    if auth_scheme.lower() != STUDIO_AUTH_SCHEME:
        return None
    if not _is_loopback_client(request):
        return None
    if not _auth_is_configured(config_auth):
        return None
    return User(identity=STUDIO_USER_ID, is_authenticated=True)


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
        if callable(obj):
            obj = obj()
        if hasattr(obj, '_authenticate_handler'):
            return LangGraphAuthBackend(obj)
        return obj
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
    auth_path = settings.AUTH_MODULE_PATH or config_auth.path
    if auth_path:
        custom_backend = _load_custom_backend(auth_path)
        if custom_backend is not None:
            _backend = custom_backend
            return _backend
    _backend = NoopAuthBackend()
    return _backend
