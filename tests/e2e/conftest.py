import os
import subprocess
import time
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import httpx
import pymysql
import pytest


def _seekdb_reachable() -> bool:
    host = os.getenv("OCEANBASE_HOST", "127.0.0.1")
    port = int(os.getenv("OCEANBASE_PORT", "2881"))
    user = os.getenv("OCEANBASE_USER", "root@test")
    password = os.getenv("OCEANBASE_PASSWORD", "")
    db_name = os.getenv("OCEANBASE_DB_NAME", "seekdb")
    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=db_name,
            autocommit=True,
            charset="utf8mb4",
        )
        conn.close()
        return True
    except Exception:  # noqa: BLE001
        return False


def _wait_for_health(base_url: str, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    with httpx.Client(timeout=2.0, trust_env=False) as client:
        while time.time() < deadline:
            try:
                response = client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return True
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
    return False


def _seekdb_url() -> str:
    host = os.getenv("OCEANBASE_HOST", "127.0.0.1")
    port = os.getenv("OCEANBASE_PORT", "2881")
    user = os.getenv("OCEANBASE_USER", "root@test").replace("@", "%40")
    password = os.getenv("OCEANBASE_PASSWORD", "")
    db_name = os.getenv("OCEANBASE_DB_NAME", "seekdb")
    return f"mysql+aiomysql://{user}:{password}@{host}:{port}/{db_name}"


def _live_provider_missing_env() -> list[str]:
    provider = os.getenv("LIVE_PROVIDER_KIND", "").strip().lower()
    if provider == "openai":
        required = [
            "LIVE_OPENAI_COMPAT_MODEL",
            "LIVE_OPENAI_COMPAT_BASE_URL",
            "LIVE_OPENAI_COMPAT_API_KEY",
        ]
    elif provider == "anthropic":
        required = [
            "LIVE_ANTHROPIC_COMPAT_MODEL",
            "LIVE_ANTHROPIC_COMPAT_BASE_URL",
            "LIVE_ANTHROPIC_COMPAT_API_KEY",
        ]
    else:
        return ["LIVE_PROVIDER_KIND"]
    return [name for name in required if not os.getenv(name, "").strip()]


def should_fail_live_provider_config() -> bool:
    live_provider_required = os.getenv("LIVE_PROVIDER_REQUIRED", "").lower() in {"1", "true", "yes"}
    return live_provider_required


def _start_e2e_server(*, graphs_path: str) -> Generator[str, None, None]:
    running_in_ci = os.getenv("CI", "").lower() in {"1", "true", "yes"}
    if not _seekdb_reachable():
        message = "SeekDB/OceanBase backend is not reachable for e2e tests."
        if running_in_ci:
            pytest.fail(message)
        pytest.skip(message)

    host = os.getenv("E2E_HOST", "127.0.0.1")
    port = int(os.getenv("E2E_PORT", "2030"))
    base_url = f"http://{host}:{port}"

    env = os.environ.copy()
    env.setdefault("SEEKDB_URL", _seekdb_url())
    env.setdefault("AGENTSEEK_GRAPHS", graphs_path)
    env.setdefault("AUTH_MODULE_PATH", "examples/auth/custom_backend.py:backend")

    log_path = Path(os.getenv("E2E_SERVER_LOG", ".tmp/e2e-server.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w+", encoding="utf-8")
    process = subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "agentseek_api.main:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    try:
        if not _wait_for_health(base_url=base_url, timeout_seconds=30.0):
            log_file.flush()
            log_file.seek(0)
            logs = log_file.read()
            message = f"E2E server failed to become healthy.\n\n{logs}"
            if running_in_ci:
                pytest.fail(message)
            pytest.skip(message)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log_file.close()


@pytest.fixture(scope="session")
def e2e_base_url() -> Generator[str, None, None]:
    graphs_path = str((Path(__file__).resolve().parent / "fixtures" / "langgraph.store-e2e.json").resolve())
    yield from _start_e2e_server(graphs_path=graphs_path)


@pytest.fixture
async def e2e_db() -> "AsyncGenerator[None, None]":
    """Initialize the in-process ``db_manager`` against the real backend.

    The e2e server runs as a subprocess, so HTTP-only tests cannot reach the
    scheduler/migration functions (the scheduler is a separate process and those
    helpers operate through the module-level ``db_manager``). This fixture wires
    the in-process ``db_manager`` to the same real SeekDB/OceanBase/MySQL backend
    so tests can invoke ``claim_due_crons``, ``dispatch_due_crons``,
    ``_apply_additive_migrations``, etc. directly and assert real-DB behavior.

    Async (not sync) so the engine's connection pool binds to pytest-asyncio's
    event loop — the same loop the ``async def`` tests run on. A sync fixture
    using ``asyncio.run`` would bind the pool to a throwaway loop and the tests
    would hit "attached to a different loop".
    """
    running_in_ci = os.getenv("CI", "").lower() in {"1", "true", "yes"}
    if not _seekdb_reachable():
        message = "SeekDB/OceanBase backend is not reachable for e2e tests."
        if running_in_ci:
            pytest.fail(message)
        pytest.skip(message)

    os.environ.setdefault("SEEKDB_URL", _seekdb_url())

    from agentseek_api.core.database import db_manager

    await db_manager.initialize()
    try:
        yield
    finally:
        await db_manager.close()


@pytest.fixture(scope="session")
def live_provider_base_url() -> Generator[str, None, None]:
    missing = _live_provider_missing_env()
    if missing:
        message = f"Live provider e2e requires configuration: {', '.join(missing)}"
        if should_fail_live_provider_config():
            pytest.fail(message)
        pytest.skip(message)

    graphs_path = str((Path(__file__).resolve().parents[2] / "examples" / "live_provider_graphs" / "manifest.json").resolve())
    yield from _start_e2e_server(graphs_path=graphs_path)
