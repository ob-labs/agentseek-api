"""Foreground launcher for an embedded SeekDB instance.

Intended to be pointed at by `SEEKDB_EMBED_CMD` in `scripts/test-seekdb.sh`.
Starts `pylibseekdb.open_with_service` on the requested port, ensures the
target database exists, then blocks until the process is killed.
"""

from __future__ import annotations

import importlib
import os
import signal
import socket
import sys
import tempfile
import threading
import time

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
    raise SystemExit(f"embedded SeekDB did not open port {port} within {timeout_seconds:.0f}s")


def _load_pylibseekdb():
    global pylibseekdb
    if pylibseekdb is not None:
        return pylibseekdb
    try:
        pylibseekdb = importlib.import_module("pylibseekdb")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Embedded SeekDB support is optional. Install it with "
            "'uv sync --dev --extra embedded' before running embedded mode."
        ) from exc
    return pylibseekdb


def main() -> int:
    host = os.environ.get("OCEANBASE_HOST", "127.0.0.1")
    port = int(os.environ.get("OCEANBASE_PORT", "2881"))
    user = os.environ.get("OCEANBASE_USER", "root")
    password = os.environ.get("OCEANBASE_PASSWORD", "")
    db_name = os.environ.get("OCEANBASE_DB_NAME", "seekdb")
    data_dir = os.environ.get("SEEKDB_EMBED_DIR") or tempfile.mkdtemp(prefix="seekdb_embed_")
    seekdb_module = _load_pylibseekdb()

    threading.Thread(
        target=seekdb_module.open_with_service,
        args=(data_dir, port),
        daemon=True,
    ).start()

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
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
