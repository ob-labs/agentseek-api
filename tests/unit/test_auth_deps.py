import pytest
from fastapi import Request

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.models.auth import User


@pytest.mark.asyncio
async def test_get_current_user_delegates_to_auth_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBackend:
        async def authenticate(self, _request: Request) -> User:
            return User(identity="from_backend", is_authenticated=True)

    monkeypatch.setattr("agentseek_api.core.auth_deps.get_auth_backend", lambda: FakeBackend())
    request = Request({"type": "http", "headers": []})
    user = await get_current_user(request)
    assert user.identity == "from_backend"
