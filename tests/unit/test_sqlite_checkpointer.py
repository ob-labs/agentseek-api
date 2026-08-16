import json
import sqlite3
from pathlib import Path

import pytest

from agentseek_api.core.sqlite_checkpointer import SqliteCheckpointSaver


def test_setup_is_idempotent_and_save_checkpoint_persists_json(tmp_path: Path) -> None:
    database_path = tmp_path / "checkpoints.sqlite3"
    saver = SqliteCheckpointSaver(url=f"sqlite+aiosqlite:///{database_path.as_posix()}")

    try:
        saver.setup()
        saver.setup()
        saver.save_checkpoint(
            thread_id="thread-unit",
            run_id="run-unit",
            payload={"messages": ["hello"], "complete": True},
        )

        with sqlite3.connect(database_path) as connection:
            row = connection.execute(
                "SELECT thread_id, run_id, checkpoint FROM agentseek_checkpoints"
            ).fetchone()
            indexes = connection.execute(
                "PRAGMA index_list(agentseek_checkpoints)"
            ).fetchall()
    finally:
        saver.close()

    assert row is not None
    assert row[:2] == ("thread-unit", "run-unit")
    assert json.loads(row[2]) == {"messages": ["hello"], "complete": True}
    assert "idx_agentseek_checkpoints_thread_run" in {index[1] for index in indexes}


def test_rejects_non_sqlite_url() -> None:
    with pytest.raises(ValueError, match="requires a SQLite URL"):
        SqliteCheckpointSaver(url="mysql+aiomysql://root@localhost/agentseek")


def test_normalizes_windows_drive_path_without_opening_database() -> None:
    saver = SqliteCheckpointSaver(url="sqlite+aiosqlite:///C:/agentseek/runtime.db")

    try:
        assert saver._engine.url.drivername == "sqlite"
        assert saver._engine.url.database == "C:/agentseek/runtime.db"
    finally:
        saver.close()
