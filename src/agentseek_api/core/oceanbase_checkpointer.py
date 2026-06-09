from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pymysql
from pymysql.connections import Connection
from sqlalchemy import text
from sqlalchemy.engine import Engine

from agentseek_api.services.sse import safe_json_dumps


class OceanBaseCheckpointSaver:
    def __init__(self, connection_args: dict[str, str]) -> None:
        self.connection_args = connection_args
        self._engine: Engine | None = None
        if "path" in connection_args:
            from pyobvector import ObVecClient  # type: ignore[import-untyped]

            client = ObVecClient(
                path=connection_args["path"],
                db_name=connection_args.get("db_name", "test"),
            )
            self._engine = client.engine

    @contextmanager
    def _connect(self):
        if self._engine is not None:
            with self._engine.connect() as conn:
                yield conn
        else:
            conn = pymysql.connect(
                host=self.connection_args["host"],
                port=int(self.connection_args["port"]),
                user=self.connection_args["user"],
                password=self.connection_args["password"],
                database=self.connection_args["db_name"],
                autocommit=True,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
            )
            try:
                yield conn
            finally:
                conn.close()

    def setup(self) -> None:
        with self._connect() as conn:
            if self._engine is not None:
                conn.execute(
                    text(
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
                )
                conn.commit()
            else:
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
            if self._engine is not None:
                conn.execute(
                    text(
                        """
                        INSERT INTO agentseek_checkpoints (thread_id, run_id, checkpoint, created_at)
                        VALUES (:thread_id, :run_id, :checkpoint, :created_at)
                        """
                    ),
                    {
                        "thread_id": thread_id,
                        "run_id": run_id,
                        "checkpoint": safe_json_dumps(payload),
                        "created_at": datetime.now(UTC).replace(tzinfo=None),
                    },
                )
                conn.commit()
            else:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO agentseek_checkpoints (thread_id, run_id, checkpoint, created_at)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (thread_id, run_id, safe_json_dumps(payload), datetime.now(UTC).replace(tzinfo=None)),
                    )
