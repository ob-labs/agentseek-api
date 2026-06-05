from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pymysql


ROOT_DIR = Path(__file__).resolve().parents[1]


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_mysql_ready(*, timeout_seconds: float) -> None:
    host = os.environ["OCEANBASE_HOST"]
    port = int(os.environ["OCEANBASE_PORT"])
    user = os.environ["OCEANBASE_USER"]
    password = os.environ["OCEANBASE_PASSWORD"]
    db_name = os.environ["OCEANBASE_DB_NAME"]
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            conn = pymysql.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                autocommit=True,
                charset="utf8mb4",
            )
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
            conn.close()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"MySQL did not become ready: {last_error}")


def _wait_for_http(url: str, *, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    with httpx.Client(timeout=2.0, trust_env=False) as client:
        while time.time() < deadline:
            try:
                response = client.get(url)
                if 200 <= response.status_code < 300:
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for '{url}': {last_error}")


def _terminate_process(process: subprocess.Popen[str], *, name: str) -> None:
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
    except Exception:  # noqa: BLE001
        process.terminate()
    try:
        process.wait(timeout=20.0)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait(timeout=10.0)
        raise RuntimeError(f"{name} did not terminate cleanly") from exc


def main() -> int:
    manifest_path = ROOT_DIR / "examples" / "minimal_agentseek" / "agentseek.json"
    tmp_dir = Path(tempfile.mkdtemp(prefix="agentseek-cli-dev-studio-smoke-"))
    api_port = _pick_free_port()
    log_path = tmp_dir / "dev.log"
    base_url = f"http://127.0.0.1:{api_port}"

    env = os.environ.copy()
    env.update(
        {
            "METADATA_DB_URL": f"sqlite+aiosqlite:///{tmp_dir / 'metadata.db'}",
            "SEEKDB_URL": (
                "mysql+aiomysql://"
                f"{env['OCEANBASE_USER']}:{env['OCEANBASE_PASSWORD']}"
                f"@{env['OCEANBASE_HOST']}:{env['OCEANBASE_PORT']}/{env['OCEANBASE_DB_NAME']}"
            ),
            "AUTH_TYPE": "api_key",
            "AUTH_API_KEYS": "secret=api-user",
        }
    )

    _wait_for_mysql_ready(timeout_seconds=120.0)

    with log_path.open("w", encoding="utf-8") as log_output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agentseek_api.cli",
                "dev",
                "--config",
                str(manifest_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(api_port),
                "--no-reload",
                "--no-browser",
            ],
            cwd=str(ROOT_DIR),
            env=env,
            stdout=log_output,
            stderr=subprocess.STDOUT,
            text=True,
        )
    try:
        _wait_for_http(f"{base_url}/health", timeout_seconds=60.0)

        with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
            docs = client.get("/docs")
            redoc = client.get("/redoc")
            openapi = client.get("/openapi.json")
            unauthorized = client.post("/assistants", json={"name": "blocked", "graph_id": "chat"})
            assistant = client.post(
                "/assistants",
                json={"name": "studio", "graph_id": "chat"},
                headers={"x-auth-scheme": "langsmith"},
            )
            thread = client.post(
                "/threads",
                json={"metadata": {"smoke": True}},
                headers={"x-auth-scheme": "langsmith"},
            )

        assert docs.status_code == 200, docs.text
        assert redoc.status_code == 200, redoc.text
        assert openapi.status_code == 200, openapi.text
        assert docs.headers["content-type"].startswith("text/html")
        assert redoc.headers["content-type"].startswith("text/html")
        assert openapi.json()["openapi"].startswith("3.")
        assert unauthorized.status_code == 401, unauthorized.text
        assert assistant.status_code == 200, assistant.text
        assert thread.status_code == 200, thread.text
        assert "thread_id" in thread.json()

        logs = log_path.read_text(encoding="utf-8", errors="replace")
        assert "> Ready!" in logs, logs
        assert f"> - API: http://localhost:{api_port}" in logs, logs
        assert f"> - Docs (Swagger): http://localhost:{api_port}/docs" in logs, logs
        assert f"> - Docs (Scalar): http://localhost:{api_port}/scalar-docs" in logs, logs
        assert (
            "https://smith.langchain.com/studio/?baseUrl="
            f"http://127.0.0.1:{api_port}"
        ) in logs, logs
    finally:
        _terminate_process(process, name="agentseek-api dev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
