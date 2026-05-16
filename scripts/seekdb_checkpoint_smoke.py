import os
import time
from uuid import uuid4

import pymysql
from pymysql.err import OperationalError

from agentseek_api.core.oceanbase_checkpointer import OceanBaseCheckpointSaver

RETRYABLE_OCEANBASE_ERROR_CODES = {
    4012,  # Timeout or service-not-ready style transient observed during startup
    4392,  # "disk is hung" reported transiently by OceanBase CE in CI startup
}

def _retry_transient_timeout(action_name: str, func, *, timeout_seconds: float = 60.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    attempt = 0
    while time.time() < deadline:
        try:
            func()
            return
        except OperationalError as exc:
            last_error = exc
            if exc.args and exc.args[0] in RETRYABLE_OCEANBASE_ERROR_CODES:
                attempt += 1
                time.sleep(min(2 * attempt, 5))
                continue
            raise
    raise RuntimeError(f"{action_name} did not succeed within {timeout_seconds:.0f}s: {last_error}") from last_error


def main() -> None:
    host = os.getenv("OCEANBASE_HOST", "127.0.0.1")
    port = os.getenv("OCEANBASE_PORT", "2881")
    user = os.getenv("OCEANBASE_USER", "root@test")
    password = os.getenv("OCEANBASE_PASSWORD", "")
    db_name = os.getenv("OCEANBASE_DB_NAME", "seekdb")

    saver = OceanBaseCheckpointSaver(
        connection_args={
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "db_name": db_name,
        }
    )
    _retry_transient_timeout("checkpoint setup", saver.setup, timeout_seconds=180.0)

    thread_id = f"smoke-thread-{uuid4()}"
    run_id = f"smoke-run-{uuid4()}"
    payload = {"input": {"message": "hello"}, "output": {"echo": "hello"}}
    _retry_transient_timeout(
        "checkpoint write",
        lambda: saver.save_checkpoint(thread_id=thread_id, run_id=run_id, payload=payload),
    )

    conn = pymysql.connect(
        host=host,
        port=int(port),
        user=user,
        password=password,
        database=db_name,
        autocommit=True,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) AS count FROM agentseek_checkpoints WHERE thread_id=%s AND run_id=%s",
                (thread_id, run_id),
            )
            row = cursor.fetchone()
            if row is None or int(row["count"]) != 1:
                raise RuntimeError("Checkpoint storage smoke test failed: checkpoint row not found")
    finally:
        conn.close()

    print("Checkpoint storage smoke test passed")


if __name__ == "__main__":
    main()
