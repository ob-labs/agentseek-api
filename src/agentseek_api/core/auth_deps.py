from fastapi import Depends, Request

from agentseek_api.core.auth_middleware import get_auth_backend
from agentseek_api.models.auth import User


async def get_current_user(request: Request) -> User:
    backend = get_auth_backend()
    return await backend.authenticate(request)


AuthDependency = Depends(get_current_user)
