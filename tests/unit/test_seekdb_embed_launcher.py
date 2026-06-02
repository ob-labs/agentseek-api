from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "seekdb_embed_launcher.py"
SPEC = importlib.util.spec_from_file_location("seekdb_embed_launcher", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
seekdb_embed_launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seekdb_embed_launcher)


def test_embed_launcher_defers_optional_pylibseekdb_import() -> None:
    assert getattr(seekdb_embed_launcher, "pylibseekdb", None) is None


def test_embed_launcher_errors_with_actionable_message_when_extra_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str) -> object:
        if name == "pylibseekdb":
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return __import__(name)

    monkeypatch.setattr(seekdb_embed_launcher.importlib, "import_module", fake_import_module)

    with pytest.raises(SystemExit, match="uv sync --dev --extra embedded"):
        seekdb_embed_launcher._load_pylibseekdb()


def test_embed_launcher_reraises_nested_module_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str) -> object:
        if name == "pylibseekdb":
            raise ModuleNotFoundError("No module named 'onnxruntime'", name="onnxruntime")
        return __import__(name)

    monkeypatch.setattr(seekdb_embed_launcher.importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError, match="onnxruntime"):
        seekdb_embed_launcher._load_pylibseekdb()


def test_embed_launcher_starts_service_and_bootstraps_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCEANBASE_DB_NAME", raising=False)
    observed: dict[str, object] = {}

    class FakeCursor:
        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def execute(self, query: str) -> None:
            observed["query"] = query

    class FakeConnection:
        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def close(self) -> None:
            observed["closed"] = True

    class FakeThread:
        def __init__(self, *, target, args, daemon: bool) -> None:
            observed["thread_target"] = target
            observed["thread_args"] = args
            observed["thread_daemon"] = daemon

        def start(self) -> None:
            observed["thread_started"] = True

    class FakeEvent:
        def __init__(self) -> None:
            self._first = True

        def set(self) -> None:
            self._first = False

        def is_set(self) -> bool:
            if self._first:
                self._first = False
                return False
            return True

    fake_pylibseekdb = SimpleNamespace(open_with_service=lambda data_dir, port: None)

    monkeypatch.setattr(seekdb_embed_launcher, "pylibseekdb", None)
    monkeypatch.setattr(
        seekdb_embed_launcher.importlib,
        "import_module",
        lambda name: fake_pylibseekdb if name == "pylibseekdb" else __import__(name),
    )
    monkeypatch.setattr(seekdb_embed_launcher.threading, "Thread", FakeThread)
    monkeypatch.setattr(seekdb_embed_launcher.threading, "Event", FakeEvent)
    monkeypatch.setattr(seekdb_embed_launcher, "_wait_for_port", lambda *args, **kwargs: None)
    monkeypatch.setattr(seekdb_embed_launcher.pymysql, "connect", lambda **kwargs: FakeConnection())
    monkeypatch.setattr(seekdb_embed_launcher.signal, "signal", lambda *args, **kwargs: None)
    monkeypatch.setattr(seekdb_embed_launcher.tempfile, "mkdtemp", lambda prefix: "/tmp/embed-dir")
    monkeypatch.setattr(seekdb_embed_launcher.time, "sleep", lambda _seconds: None)

    assert seekdb_embed_launcher.main() == 0
    assert observed["thread_started"] is True
    assert observed["thread_daemon"] is True
    assert observed["thread_args"] == ("/tmp/embed-dir", 2881)
    assert "CREATE DATABASE IF NOT EXISTS `seekdb`" == observed["query"]
    assert observed["closed"] is True
