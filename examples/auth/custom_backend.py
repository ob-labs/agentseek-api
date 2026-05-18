from fastapi import Request

from agentseek_api.models.auth import User


class HeaderAuthBackend:
    async def authenticate(self, request: Request) -> User:
        identity = request.headers.get("x-user-id", "example-user")
        return User(identity=identity, is_authenticated=True)


backend = HeaderAuthBackend
