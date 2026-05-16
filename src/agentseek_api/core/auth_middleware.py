from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Protocol

from fastapi import Request

from agentseek_api.models.auth import User
from agentseek_api.settings import settings


class AuthBackend(Protocol):
    async def authenticate(self, request: Request) -> User: ...


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
    spec.loader.exec_module(module)
    return module


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
