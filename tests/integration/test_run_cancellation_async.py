from pathlib import Path
from unittest.mock import patch
import time

from fastapi import Request
from fastapi.testclient import TestClient

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.main import create_app
from agentseek_api.models.auth import User
from agentseek_api.settings import settings


class FakeCheckpointer:
    def __init__(self, connection_args: dict[str, str]) -> None:
        self.connection_args = connection_args

    def setup(self) -> None:
        return None

    def save_checkpoint(self, **_kwargs: object) -> None:
        return None


async def header_user_override(request: Request) -> User:
    identity = request.headers.get("x-user-id", "default_user")
    return User(identity=identity, is_authenticated=True)


def test_cancelled_run_is_not_overwritten_by_background_completion(tmp_path: Path) -> None:
    db_path = tmp_path / "cancel-race.db"
    with patch("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer):
        settings.SEEKDB_URL = f"sqlite+aiosqlite:///{db_path}"
        app = create_app()
        app.dependency_overrides[get_current_user] = header_user_override
        with TestClient(app) as client:
            assistant = client.post("/assistants", json={"name": "stress", "graph_id": "stress_test"})
            assert assistant.status_code == 200
            assistant_id = assistant.json()["assistant_id"]

            thread = client.post("/threads", json={"metadata": {}}, headers={"x-user-id": "u1"})
            assert thread.status_code == 200
            thread_id = thread.json()["thread_id"]

            created = client.post(
                f"/threads/{thread_id}/runs",
                json={"assistant_id": assistant_id, "input": {"delay": 0.05, "steps": 20}},
                headers={"x-user-id": "u1"},
            )
            assert created.status_code == 200
            run_id = created.json()["run_id"]

            cancelled = client.post(
                f"/threads/{thread_id}/runs/{run_id}/cancel",
                headers={"x-user-id": "u1"},
            )
            assert cancelled.status_code == 200

            time.sleep(2)

            fetched = client.get(
                f"/threads/{thread_id}/runs/{run_id}",
                headers={"x-user-id": "u1"},
            )
            assert fetched.status_code == 200
            assert fetched.json()["status"] == "error"
            assert fetched.json()["last_error"] == "Run cancelled"
            assert fetched.json()["output"] is None

            state = client.get(
                f"/threads/{thread_id}/state",
                headers={"x-user-id": "u1"},
            )
            assert state.status_code == 200
            assert state.json()["values"] == {}

            history = client.get(
                f"/threads/{thread_id}/history",
                headers={"x-user-id": "u1"},
            )
            assert history.status_code == 200
            assert history.json() == []
