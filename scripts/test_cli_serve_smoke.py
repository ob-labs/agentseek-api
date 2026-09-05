from __future__ import annotations

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


def _wait_for_log_line(log_path: Path, needle: str, *, timeout_seconds: float, process: subprocess.Popen[str]) -> None:
    deadline = time.time() + timeout_seconds
    last_content = ""
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Process exited before writing '{needle}' to {log_path}:\n{last_content}")
        if log_path.exists():
            last_content = log_path.read_text(encoding="utf-8", errors="replace")
            if needle in last_content:
                return
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for '{needle}' in {log_path}:\n{last_content}")


def _wait_for_http_json(url: str, *, timeout_seconds: float, method: str = "GET", data: bytes | None = None) -> str:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            req = urllib_request.Request(url, method=method, data=data)
            if data is not None:
                req.add_header("Content-Type", "application/json")
            with urllib_request.urlopen(req, timeout=2.0) as response:
                if 200 <= response.status < 300:
                    return response.read().decode("utf-8")
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(1.0)
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
    manifest_path = ROOT_DIR / "examples" / "sample_graphs_manifest.json"
    tmp_dir = Path(tempfile.mkdtemp(prefix="agentseek-cli-serve-smoke-"))
    api_port = _pick_free_port()
    seekdb_port = _pick_free_port()
    seekdb_log = tmp_dir / "seekdb.log"
    serve_log = tmp_dir / "serve.log"

    env = os.environ.copy()
    env.update(
        {
            "METADATA_DB_URL": f"sqlite+aiosqlite:///{tmp_dir / 'metadata.db'}",
            "OCEANBASE_HOST": "127.0.0.1",
            "OCEANBASE_PORT": str(seekdb_port),
            "OCEANBASE_USER": "root",
            "OCEANBASE_PASSWORD": "",
            "OCEANBASE_DB_NAME": "seekdb",
            "SEEKDB_URL": f"mysql+aiomysql://root:@127.0.0.1:{seekdb_port}/seekdb",
        }
    )

    with seekdb_log.open("w", encoding="utf-8") as seekdb_output:
        seekdb_process = subprocess.Popen(
            [sys.executable, str(ROOT_DIR / "scripts" / "seekdb_embed_launcher.py")],
            cwd=str(ROOT_DIR),
            env=env,
            stdout=seekdb_output,
            stderr=subprocess.STDOUT,
            text=True,
        )
    try:
        _wait_for_log_line(
            seekdb_log,
            "embedded seekdb listening",
            timeout_seconds=60.0,
            process=seekdb_process,
        )

        with serve_log.open("w", encoding="utf-8") as serve_output:
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
                stdout=serve_output,
                stderr=subprocess.STDOUT,
                text=True,
            )
        try:
            health_body = _wait_for_http_json(
                f"http://127.0.0.1:{api_port}/health",
                timeout_seconds=60.0,
            )
            info_body = _wait_for_http_json(
                f"http://127.0.0.1:{api_port}/info",
                timeout_seconds=30.0,
            )
            assistants_body = _wait_for_http_json(
                f"http://127.0.0.1:{api_port}/assistants/search",
                timeout_seconds=30.0,
                method="POST",
                data=b"{}",
            )
            assert '"healthy"' in health_body
            assert '"flags"' in info_body
            assert assistants_body.strip().startswith("[")
        finally:
            _terminate_process(serve_process, name="agentseek-api serve")
    finally:
        _terminate_process(seekdb_process, name="embedded seekdb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
