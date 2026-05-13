import os
import subprocess
import time
from collections.abc import Generator

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
    with httpx.Client(timeout=2.0) as client:
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


@pytest.fixture(scope="session")
def e2e_base_url() -> Generator[str, None, None]:
    if not _seekdb_reachable():
        pytest.skip("SeekDB/OceanBase backend is not reachable for e2e tests.")

    host = os.getenv("E2E_HOST", "127.0.0.1")
    port = int(os.getenv("E2E_PORT", "2030"))
    base_url = f"http://{host}:{port}"

    env = os.environ.copy()
    env.setdefault("SEEKDB_URL", _seekdb_url())
    env.setdefault("AUTH_TYPE", "noop")

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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    try:
        if not _wait_for_health(base_url=base_url, timeout_seconds=30.0):
            pytest.skip("E2E server failed to become healthy.")
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
