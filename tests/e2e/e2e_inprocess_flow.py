from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fastapi import Request
from fastapi.testclient import TestClient

import agentseek_api.core.database as database_module
import agentseek_api.services.run_preparation as run_preparation_module
from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.main import create_app
from agentseek_api.models.auth import User
from agentseek_api.settings import settings
from agentseek_api.services.run_executor import RunExecutionResult


class FakeCheckpointer:
    def __init__(self, connection_args: dict[str, str]) -> None:
        self.connection_args = connection_args

    def setup(self) -> None:
        return None

    def save_checkpoint(self, *, thread_id: str, run_id: str, payload: dict[str, Any]) -> None:
        _ = (thread_id, run_id, payload)


class InlineExecutor:
    async def submit(self, func: Callable[[], Awaitable[None]]) -> None:
        await func()


async def fake_execute_run(
    *,
    thread_id: str,
    run_id: str,
    payload: dict[str, Any],
    graph_id: str | None = None,
    resume: Any = None,
) -> RunExecutionResult:
    _ = resume
    return RunExecutionResult(
        output={"echo": payload, "thread_id": thread_id, "run_id": run_id, "graph_id": graph_id},
        interrupted=False,
        interrupts=[],
    )


async def header_user(request: Request) -> User:
    identity = request.headers.get("x-user-id", "example-user")
    return User(identity=identity, is_authenticated=True)


def main() -> None:
    with TemporaryDirectory(prefix="agentseek-example-") as tmpdir:
        settings.SEEKDB_URL = f"sqlite+aiosqlite:///{Path(tmpdir) / 'metadata.db'}"
        database_module.OceanBaseCheckpointSaver = FakeCheckpointer
        run_preparation_module.execute_run = fake_execute_run
        run_preparation_module.get_executor = lambda: InlineExecutor()

        app = create_app()
        app.dependency_overrides[get_current_user] = header_user

        with TestClient(app) as client:
            assistant = client.post("/assistants", json={"name": "example-assistant", "graph_id": "default"})
            assert assistant.status_code == 200, assistant.text
            assistant_id = assistant.json()["assistant_id"]

            thread = client.post("/threads", json={"metadata": {"source": "example"}}, headers={"x-user-id": "example-user"})
            assert thread.status_code == 200, thread.text
            thread_id = thread.json()["thread_id"]

            run = client.post(
                f"/threads/{thread_id}/runs",
                json={"assistant_id": assistant_id, "input": {"message": "hello"}},
                headers={"x-user-id": "example-user"},
            )
            assert run.status_code == 200, run.text
            run_id = run.json()["run_id"]

            waited = client.get(f"/threads/{thread_id}/runs/{run_id}/wait", headers={"x-user-id": "example-user"})
            assert waited.status_code == 200, waited.text
            assert waited.json()["status"] == "success"

            streamed = client.get(f"/threads/{thread_id}/runs/{run_id}/stream", headers={"x-user-id": "example-user"})
            assert streamed.status_code == 200, streamed.text
            assert "event: end" in streamed.text

            stateless = client.post(
                "/runs",
                json={"assistant_id": assistant_id, "input": {"mode": "stateless"}},
                headers={"x-user-id": "example-user"},
            )
            assert stateless.status_code == 200, stateless.text

    print("In-process end-to-end flow passed")


if __name__ == "__main__":
    main()
