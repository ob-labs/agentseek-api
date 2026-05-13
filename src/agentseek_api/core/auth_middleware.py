from importlib import import_module
from typing import Protocol

from fastapi import Request

from agentseek_api.models.auth import User
from agentseek_api.settings import settings


class AuthBackend(Protocol):
    async def authenticate(self, request: Request) -> User: ...


class NoopAuthBackend:
    async def authenticate(self, _request: Request) -> User:
        return User(identity="default_user", is_authenticated=True)


def _load_custom_backend() -> AuthBackend | None:
    auth_module_path = settings.AUTH_MODULE_PATH
    if not auth_module_path:
        return None
    if ":" not in auth_module_path:
        raise RuntimeError(
            f"Invalid AUTH_MODULE_PATH='{auth_module_path}'. Expected format 'module.path:symbol'."
        )

    module_name, symbol = auth_module_path.split(":", maxsplit=1)
    if not module_name or not symbol:
        raise RuntimeError(
            f"Invalid AUTH_MODULE_PATH='{auth_module_path}'. Expected format 'module.path:symbol'."
        )

    try:
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

    auth_type = settings.AUTH_TYPE.strip().lower()
    if auth_type == "noop":
        _backend = NoopAuthBackend()
        return _backend
    if auth_type == "custom":
        custom_backend = _load_custom_backend()
        if custom_backend is None:
            raise RuntimeError("AUTH_TYPE=custom requires AUTH_MODULE_PATH to be configured.")
        _backend = custom_backend
        return _backend
    raise ValueError(f"Unsupported AUTH_TYPE: {settings.AUTH_TYPE}")

    return _backend
