import os
from uuid import uuid4

import pymysql

from agentseek_api.core.oceanbase_checkpointer import OceanBaseCheckpointSaver


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
    saver.setup()

    thread_id = f"smoke-thread-{uuid4()}"
    run_id = f"smoke-run-{uuid4()}"
    payload = {"input": {"message": "hello"}, "output": {"echo": "hello"}}
    saver.save_checkpoint(thread_id=thread_id, run_id=run_id, payload=payload)

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
