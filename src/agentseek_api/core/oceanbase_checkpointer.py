import json
from datetime import UTC, datetime
from typing import Any

import pymysql
from pymysql.connections import Connection


class OceanBaseCheckpointSaver:
    def __init__(self, connection_args: dict[str, str]) -> None:
        self.connection_args = connection_args

    def _connect(self) -> Connection:
        return pymysql.connect(
            host=self.connection_args["host"],
            port=int(self.connection_args["port"]),
            user=self.connection_args["user"],
            password=self.connection_args["password"],
            database=self.connection_args["db_name"],
            autocommit=True,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )

    def setup(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agentseek_checkpoints (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        thread_id VARCHAR(255) NOT NULL,
                        run_id VARCHAR(255) NOT NULL,
                        checkpoint JSON NOT NULL,
                        created_at DATETIME NOT NULL,
                        INDEX idx_thread_run (thread_id, run_id)
                    )
                    """
                )

    def save_checkpoint(self, *, thread_id: str, run_id: str, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO agentseek_checkpoints (thread_id, run_id, checkpoint, created_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (thread_id, run_id, json.dumps(payload, default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o)), datetime.now(UTC).replace(tzinfo=None)),
                )
