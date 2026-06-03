from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentseek_api.main import _include_custom_routers


def _make_custom_app_file(tmp_path: Path) -> Path:
    f = tmp_path / "routes.py"
    f.write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "@app.get('/custom/ping')\n"
        "def ping(): return {'pong': True}\n",
        encoding="utf-8",
    )
    return f


def test_include_custom_routers_adds_route(tmp_path: Path) -> None:
    f = _make_custom_app_file(tmp_path)
    http_config = {"app": f"{f}:app"}

    app = FastAPI()

    with patch("agentseek_api.main.get_http_config", return_value=http_config), \
         patch("agentseek_api.main.get_config_dir", return_value=tmp_path):
        _include_custom_routers(app)

    client = TestClient(app)
    resp = client.get("/custom/ping")
    assert resp.status_code == 200
    assert resp.json() == {"pong": True}


def test_include_custom_routers_skips_default_paths(tmp_path: Path) -> None:
    f = tmp_path / "with_docs.py"
    f.write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "@app.get('/custom/hello')\n"
        "def hello(): return 'hi'\n",
        encoding="utf-8",
    )
    http_config = {"app": f"{f}:app"}

    app = FastAPI()
    initial_route_count = len(app.routes)

    with patch("agentseek_api.main.get_http_config", return_value=http_config), \
         patch("agentseek_api.main.get_config_dir", return_value=tmp_path):
        _include_custom_routers(app)

    paths = [getattr(r, "path", None) for r in app.router.routes]
    assert "/custom/hello" in paths
    assert "/docs" not in paths[initial_route_count:]


def test_include_custom_routers_skips_conflicting_paths(tmp_path: Path) -> None:
    f = tmp_path / "conflict.py"
    f.write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "@app.get('/existing')\n"
        "def conflict(): return 'shadow'\n",
        encoding="utf-8",
    )
    http_config = {"app": f"{f}:app"}

    app = FastAPI()

    @app.get("/existing")
    def existing():
        return "original"

    with patch("agentseek_api.main.get_http_config", return_value=http_config), \
         patch("agentseek_api.main.get_config_dir", return_value=tmp_path):
        _include_custom_routers(app)

    client = TestClient(app)
    resp = client.get("/existing")
    assert resp.json() == "original"


def test_include_custom_routers_noop_without_config() -> None:
    app = FastAPI()
    route_count_before = len(app.routes)

    with patch("agentseek_api.main.get_http_config", return_value=None):
        _include_custom_routers(app)

    assert len(app.routes) == route_count_before


def test_include_custom_routers_raises_on_bad_app(tmp_path: Path) -> None:
    import pytest

    http_config = {"app": f"{tmp_path}/nonexistent.py:app"}

    app = FastAPI()

    with patch("agentseek_api.main.get_http_config", return_value=http_config), \
         patch("agentseek_api.main.get_config_dir", return_value=tmp_path), \
         pytest.raises(FileNotFoundError):
        _include_custom_routers(app)
