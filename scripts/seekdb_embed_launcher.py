"""Foreground launcher for an embedded seekdb instance.

Intended to be pointed at by `SEEKDB_EMBED_CMD` in `scripts/test-seekdb.sh`.
Starts the seekdb server bundled with `pylibseekdb` on the requested port,
ensures the target database exists, then blocks until the process is killed.
"""

from __future__ import annotations

import importlib
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pymysql


pylibseekdb = None


def _wait_for_port(host: str, port: int, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.3)
    raise SystemExit(f"embedded seekdb did not open port {port} within {timeout_seconds:.0f}s")


def _load_pylibseekdb():
    global pylibseekdb
    if pylibseekdb is not None:
        return pylibseekdb
    try:
        pylibseekdb = importlib.import_module("pylibseekdb")
    except ModuleNotFoundError as exc:
        if exc.name != "pylibseekdb":
            raise
        raise SystemExit(
            "Embedded seekdb support is optional. Install it with "
            "'uv sync --dev --extra embedded' before running embedded mode."
        ) from exc
    return pylibseekdb


def _resolve_seekdb_binary(seekdb_module) -> Path:
    filename = "seekdb.exe" if os.name == "nt" else "seekdb"
    binary = Path(seekdb_module.__file__).resolve().with_name(filename)
    if not binary.is_file():
        raise SystemExit(f"Bundled seekdb server binary was not found at {binary}")
    return binary


def main() -> int:
    host = os.environ.get("OCEANBASE_HOST", "127.0.0.1")
    port = int(os.environ.get("OCEANBASE_PORT", "2881"))
    user = os.environ.get("OCEANBASE_USER", "root")
    password = os.environ.get("OCEANBASE_PASSWORD", "")
    db_name = os.environ.get("OCEANBASE_DB_NAME", "seekdb")
    data_dir = os.environ.get("SEEKDB_EMBED_DIR") or tempfile.mkdtemp(prefix="seekdb_embed_")
    seekdb_module = _load_pylibseekdb()
    server_binary = _resolve_seekdb_binary(seekdb_module)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    server_process = subprocess.Popen(
        [
            str(server_binary),
            "--nodaemon",
            "--port",
            str(port),
            "--base-dir",
            data_dir,
        ]
    )
    try:
        _wait_for_port(host, port, 30.0)

        bootstrap_user = user.split("@", 1)[0] if "@" in user else user
        conn = pymysql.connect(
            host=host,
            port=port,
            user=bootstrap_user,
            password=password,
            autocommit=True,
            charset="utf8mb4",
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
        finally:
            conn.close()

        print(f"embedded seekdb listening host={host} port={port} db={db_name} dir={data_dir}", flush=True)

        stop = threading.Event()

        def _handle_signal(signum, _frame):
            stop.set()

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)
        while not stop.is_set():
            returncode = server_process.poll()
            if returncode is not None:
                raise SystemExit(f"embedded seekdb server exited with status {returncode}")
            time.sleep(1.0)
        return 0
    finally:
        if server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait(timeout=10.0)


if __name__ == "__main__":
    sys.exit(main())
