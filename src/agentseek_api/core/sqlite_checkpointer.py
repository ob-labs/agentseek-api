from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.pool import StaticPool

from agentseek_api.services.sse import safe_json_dumps


class SqliteCheckpointSaver:
    """Persist completed run snapshots in the configured SQLite database."""

    def __init__(self, *, url: str) -> None:
        parsed_url = make_url(url)
        if parsed_url.get_backend_name() != "sqlite":
            raise ValueError("SqliteCheckpointSaver requires a SQLite URL")

        sqlite_url = parsed_url.set(drivername="sqlite").render_as_string(
            hide_password=False
        )
        engine_options: dict[str, Any] = {"connect_args": {"check_same_thread": False}}
        if parsed_url.database in {None, "", ":memory:"}:
            engine_options["poolclass"] = StaticPool

        self._engine: Engine = create_engine(sqlite_url, **engine_options)
        self._lock = threading.Lock()

    def setup(self) -> None:
        with self._lock, self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS agentseek_checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        thread_id TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        checkpoint TEXT NOT NULL,
                        created_at DATETIME NOT NULL
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_agentseek_checkpoints_thread_run
                    ON agentseek_checkpoints (thread_id, run_id)
                    """
                )
            )

    def save_checkpoint(
        self,
        *,
        thread_id: str,
        run_id: str,
        payload: dict[str, Any],
    ) -> None:
        with self._lock, self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO agentseek_checkpoints
                        (thread_id, run_id, checkpoint, created_at)
                    VALUES (:thread_id, :run_id, :checkpoint, :created_at)
                    """
                ),
                {
                    "thread_id": thread_id,
                    "run_id": run_id,
                    "checkpoint": safe_json_dumps(payload),
                    "created_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                },
            )

    def close(self) -> None:
        with self._lock:
            self._engine.dispose()
