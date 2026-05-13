from typing import Any

import pytest

from agentseek_api.core.oceanbase_checkpointer import OceanBaseCheckpointSaver


class FakeCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...] | None]] = []

    def execute(self, query: str, params: tuple[Any, ...] | None = None) -> None:
        self.calls.append((query, params))

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        return None


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self._cursor

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()


def test_setup_executes_create_table(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cursor = FakeCursor()
    fake_conn = FakeConnection(fake_cursor)
    monkeypatch.setattr("agentseek_api.core.oceanbase_checkpointer.pymysql.connect", lambda **_kwargs: fake_conn)

    saver = OceanBaseCheckpointSaver(
        connection_args={"host": "h", "port": "2881", "user": "u", "password": "p", "db_name": "seekdb"}
    )
    saver.setup()

    assert fake_cursor.calls
    assert "CREATE TABLE IF NOT EXISTS agentseek_checkpoints" in fake_cursor.calls[0][0]


def test_save_checkpoint_executes_insert(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cursor = FakeCursor()
    fake_conn = FakeConnection(fake_cursor)
    monkeypatch.setattr("agentseek_api.core.oceanbase_checkpointer.pymysql.connect", lambda **_kwargs: fake_conn)

    saver = OceanBaseCheckpointSaver(
        connection_args={"host": "h", "port": "2881", "user": "u", "password": "p", "db_name": "seekdb"}
    )
    saver.save_checkpoint(thread_id="t1", run_id="r1", payload={"hello": "world"})

    assert fake_cursor.calls
    query, params = fake_cursor.calls[0]
    assert "INSERT INTO agentseek_checkpoints" in query
    assert params is not None
    assert params[0] == "t1"
    assert params[1] == "r1"
