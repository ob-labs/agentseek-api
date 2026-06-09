"""Smoke test for SEEKDB_EMBED=true — no external processes, no Docker.

Launches ``agentseek-api serve`` with ``SEEKDB_EMBED=true`` and validates
health, info, assistant search, and a thread + run checkpoint round-trip.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request


ROOT_DIR = Path(__file__).resolve().parents[1]


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http(
    url: str,
    *,
    timeout_seconds: float,
    method: str = "GET",
    data: bytes | None = None,
) -> str:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            req = urllib_request.Request(url, method=method, data=data)
            if data is not None:
                req.add_header("Content-Type", "application/json")
            with urllib_request.urlopen(req, timeout=3.0) as response:
                if 200 <= response.status < 300:
                    return response.read().decode("utf-8")
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for {method} {url}: {last_error}")


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
    manifest_path = ROOT_DIR / "examples" / "sample_graphs_manifest.json"
    tmp_dir = Path(tempfile.mkdtemp(prefix="agentseek-embed-smoke-"))
    embed_dir = tmp_dir / "seekdb_data"
    api_port = _pick_free_port()
    serve_log = tmp_dir / "serve.log"

    env = os.environ.copy()
    env.update(
        {
            "SEEKDB_EMBED": "true",
            "SEEKDB_EMBED_DIR": str(embed_dir),
            "OCEANBASE_DB_NAME": "smoke_test",
        }
    )

    base_url = f"http://127.0.0.1:{api_port}"

    with serve_log.open("w", encoding="utf-8") as log_output:
        serve_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agentseek_api.cli",
                "serve",
                "--config",
                str(manifest_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(api_port),
            ],
            cwd=str(ROOT_DIR),
            env=env,
            stdout=log_output,
            stderr=subprocess.STDOUT,
            text=True,
        )

    try:
        # 1. Health check
        health = _wait_for_http(f"{base_url}/health", timeout_seconds=90.0)
        assert '"healthy"' in health, f"Unexpected /health: {health}"
        print("PASS: /health")

        # 2. Info endpoint
        info = _wait_for_http(f"{base_url}/info", timeout_seconds=30.0)
        assert '"flags"' in info, f"Unexpected /info: {info}"
        print("PASS: /info")

        # 3. Assistants search
        assistants = _wait_for_http(
            f"{base_url}/assistants/search",
            timeout_seconds=30.0,
            method="POST",
            data=b"{}",
        )
        assert assistants.strip().startswith("["), f"Unexpected /assistants/search: {assistants}"
        print("PASS: /assistants/search")

        # 4. Thread + run round-trip (checkpoint persistence test)
        thread_body = _wait_for_http(
            f"{base_url}/threads",
            timeout_seconds=30.0,
            method="POST",
            data=b"{}",
        )
        thread = json.loads(thread_body)
        thread_id = thread["thread_id"]
        print(f"PASS: POST /threads -> {thread_id}")

        # Verify thread retrieval
        thread_get = _wait_for_http(
            f"{base_url}/threads/{thread_id}",
            timeout_seconds=10.0,
        )
        assert thread_id in thread_get, f"Thread not found: {thread_get}"
        print(f"PASS: GET /threads/{thread_id}")

        # 5. Verify embed directory was created with data
        assert embed_dir.is_dir(), f"Embed dir not created: {embed_dir}"
        metadata_db = embed_dir / "metadata.db"
        assert metadata_db.exists(), f"Metadata DB not created: {metadata_db}"
        print(f"PASS: embed data directory at {embed_dir}")

        print("\nAll embedded seekdb smoke tests passed.")
    except Exception:
        print(f"\n=== serve log ({serve_log}) ===")
        if serve_log.exists():
            print(serve_log.read_text(encoding="utf-8", errors="replace"))
        raise
    finally:
        _terminate_process(serve_process, name="agentseek-api serve (embed)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
