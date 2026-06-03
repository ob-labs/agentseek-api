from pathlib import Path

import pytest
from fastapi import FastAPI

from agentseek_api.core.app_loader import load_custom_app


def _write_custom_app(tmp_path: Path, varname: str = "app") -> Path:
    f = tmp_path / "custom.py"
    f.write_text(
        f"from fastapi import FastAPI\n{varname} = FastAPI()\n@{varname}.get('/hello')\ndef hello(): return 'hi'\n",
        encoding="utf-8",
    )
    return f


def test_load_file_based_app(tmp_path: Path) -> None:
    f = _write_custom_app(tmp_path)
    app = load_custom_app(f"{f}:app")
    assert isinstance(app, FastAPI)


def test_load_relative_path_with_base_dir(tmp_path: Path) -> None:
    _write_custom_app(tmp_path)
    app = load_custom_app("./custom.py:app", base_dir=tmp_path)
    assert isinstance(app, FastAPI)


def test_invalid_format_missing_colon() -> None:
    with pytest.raises(ValueError, match="Invalid app import path format"):
        load_custom_app("no_colon_here")


def test_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_custom_app(f"{tmp_path}/nonexistent.py:app")


def test_attribute_not_found(tmp_path: Path) -> None:
    f = _write_custom_app(tmp_path, varname="app")
    with pytest.raises(AttributeError, match="not found in module"):
        load_custom_app(f"{f}:missing_var")


def test_non_fastapi_object(tmp_path: Path) -> None:
    f = tmp_path / "bad.py"
    f.write_text("app = 'not a fastapi app'\n", encoding="utf-8")
    with pytest.raises(TypeError, match="not a FastAPI application"):
        load_custom_app(f"{f}:app")


def test_module_based_import_non_fastapi_raises() -> None:
    with pytest.raises(TypeError, match="not a FastAPI application"):
        load_custom_app("fastapi:FastAPI")
