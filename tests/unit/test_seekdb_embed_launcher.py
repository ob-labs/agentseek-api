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


def test_embed_launcher_starts_bundled_server_and_bootstraps_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OCEANBASE_DB_NAME", raising=False)
    observed: dict[str, object] = {}

    package_dir = tmp_path / "pylibseekdb"
    package_dir.mkdir()
    package_init = package_dir / "__init__.py"
    package_init.touch()
    server_binary = package_dir / (
        "seekdb.exe" if seekdb_embed_launcher.os.name == "nt" else "seekdb"
    )
    server_binary.touch()

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

    class FakeProcess:
        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            observed["terminated"] = True

        def wait(self, *, timeout: float) -> int:
            observed["wait_timeout"] = timeout
            return 0

        def kill(self) -> None:
            observed["killed"] = True

    def fake_popen(command: list[str]) -> FakeProcess:
        observed["command"] = command
        return FakeProcess()

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

    fake_pylibseekdb = SimpleNamespace(__file__=str(package_init))

    monkeypatch.setattr(seekdb_embed_launcher, "pylibseekdb", None)
    monkeypatch.setattr(
        seekdb_embed_launcher.importlib,
        "import_module",
        lambda name: fake_pylibseekdb if name == "pylibseekdb" else __import__(name),
    )
    monkeypatch.setattr(seekdb_embed_launcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(seekdb_embed_launcher.threading, "Event", FakeEvent)
    monkeypatch.setattr(seekdb_embed_launcher, "_wait_for_port", lambda *args, **kwargs: None)
    monkeypatch.setattr(seekdb_embed_launcher.pymysql, "connect", lambda **kwargs: FakeConnection())
    monkeypatch.setattr(seekdb_embed_launcher.signal, "signal", lambda *args, **kwargs: None)
    monkeypatch.setattr(seekdb_embed_launcher.tempfile, "mkdtemp", lambda prefix: "/tmp/embed-dir")
    monkeypatch.setattr(seekdb_embed_launcher.time, "sleep", lambda _seconds: None)

    assert seekdb_embed_launcher.main() == 0
    assert observed["command"] == [
        str(server_binary),
        "--nodaemon",
        "--port",
        "2881",
        "--base-dir",
        "/tmp/embed-dir",
    ]
    assert "CREATE DATABASE IF NOT EXISTS `seekdb`" == observed["query"]
    assert observed["closed"] is True
    assert observed["terminated"] is True
    assert observed["wait_timeout"] == 20.0
