from typing import Any

from fastapi import Depends, HTTPException, Request
from sqlalchemy.sql import Select

from agentseek_api.core.auth_middleware import (
    LangGraphAuthBackend,
    get_auth_backend,
    get_studio_user,
)
from agentseek_api.models.auth import User


async def get_current_user(request: Request) -> User:
    studio_user = get_studio_user(request)
    if studio_user is not None:
        return studio_user
    backend = get_auth_backend()
    user = await backend.authenticate(request)
    if not user.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


async def authorize(
    user: User,
    resource: str,
    action: str,
    value: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    backend = get_auth_backend()
    if not isinstance(backend, LangGraphAuthBackend):
        return None
    if value is None:
        value = {}
    try:
        result = await backend.authorize(user, resource, action, value)
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and 400 <= status < 500:
            detail = getattr(exc, "detail", "Forbidden")
            raise HTTPException(status_code=status, detail=detail)
        raise
    if isinstance(result, dict):
        return result
    return None


def apply_metadata_filters(stmt: Select, model: Any, filters: dict[str, Any] | None) -> Select:
    if not filters:
        return stmt
    for key, condition in filters.items():
        if isinstance(condition, dict):
            if "$eq" in condition:
                stmt = stmt.where(model.metadata_json[key].as_string() == str(condition["$eq"]))
            elif "$contains" in condition:
                contains_val = condition["$contains"]
                if isinstance(contains_val, list):
                    for item in contains_val:
                        stmt = stmt.where(model.metadata_json[key].as_string().contains(str(item)))
                else:
                    stmt = stmt.where(model.metadata_json[key].as_string().contains(str(contains_val)))
        else:
            stmt = stmt.where(model.metadata_json[key].as_string() == str(condition))
    return stmt


AuthDependency = Depends(get_current_user)
