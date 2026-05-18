from fastapi import Depends, HTTPException, Request

from agentseek_api.core.auth_middleware import get_auth_backend
from agentseek_api.models.auth import User


async def get_current_user(request: Request) -> User:
    backend = get_auth_backend()
    user = await backend.authenticate(request)
    if not user.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


AuthDependency = Depends(get_current_user)
