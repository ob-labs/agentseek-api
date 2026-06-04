from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentseek_api.main import _merge_custom_app


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


def test_merge_custom_app_adds_route(tmp_path: Path) -> None:
    f = _make_custom_app_file(tmp_path)
    http_config = {"app": f"{f}:app"}

    app = FastAPI()

    with patch("agentseek_api.main.get_http_config", return_value=http_config), \
         patch("agentseek_api.main.get_config_dir", return_value=tmp_path):
        _merge_custom_app(app)

    client = TestClient(app)
    resp = client.get("/custom/ping")
    assert resp.status_code == 200
    assert resp.json() == {"pong": True}


def test_merge_custom_app_skips_default_paths(tmp_path: Path) -> None:
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
        _merge_custom_app(app)

    paths = [getattr(r, "path", None) for r in app.router.routes]
    assert "/custom/hello" in paths
    assert "/docs" not in paths[initial_route_count:]


def test_merge_custom_app_skips_conflicting_paths(tmp_path: Path) -> None:
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
        _merge_custom_app(app)

    client = TestClient(app)
    resp = client.get("/existing")
    assert resp.json() == "original"


def test_merge_custom_app_noop_without_config() -> None:
    app = FastAPI()
    route_count_before = len(app.routes)

    with patch("agentseek_api.main.get_http_config", return_value=None):
        _merge_custom_app(app)

    assert len(app.routes) == route_count_before


def test_merge_custom_app_raises_on_bad_app(tmp_path: Path) -> None:
    import pytest

    http_config = {"app": f"{tmp_path}/nonexistent.py:app"}

    app = FastAPI()

    with patch("agentseek_api.main.get_http_config", return_value=http_config), \
         patch("agentseek_api.main.get_config_dir", return_value=tmp_path), \
         pytest.raises(FileNotFoundError):
        _merge_custom_app(app)


def test_merge_custom_app_merges_lifespan(tmp_path: Path) -> None:
    f = tmp_path / "with_lifespan.py"
    f.write_text(
        "from contextlib import asynccontextmanager\n"
        "from collections.abc import AsyncIterator\n"
        "from fastapi import FastAPI\n"
        "@asynccontextmanager\n"
        "async def custom_lifespan(app: FastAPI) -> AsyncIterator[None]:\n"
        "    app.state.custom_started = True\n"
        "    yield\n"
        "    app.state.custom_stopped = True\n"
        "app = FastAPI(lifespan=custom_lifespan)\n"
        "@app.get('/custom/ls')\n"
        "def ls(): return {'ok': True}\n",
        encoding="utf-8",
    )
    http_config = {"app": f"{f}:app"}

    from contextlib import asynccontextmanager as acm
    from collections.abc import AsyncIterator

    core_started = False
    core_stopped = False

    @acm
    async def core_lifespan(a: FastAPI) -> AsyncIterator[None]:
        nonlocal core_started, core_stopped
        core_started = True
        yield
        core_stopped = True

    app = FastAPI(lifespan=core_lifespan)

    with patch("agentseek_api.main.get_http_config", return_value=http_config), \
         patch("agentseek_api.main.get_config_dir", return_value=tmp_path):
        _merge_custom_app(app)

    with TestClient(app) as client:
        assert core_started
        assert getattr(app.state, "custom_started", False)
        resp = client.get("/custom/ls")
        assert resp.status_code == 200

    assert core_stopped
    assert getattr(app.state, "custom_stopped", False)


def test_merge_custom_app_merges_middleware(tmp_path: Path) -> None:
    f = tmp_path / "with_middleware.py"
    f.write_text(
        "from fastapi import FastAPI\n"
        "from starlette.middleware.base import BaseHTTPMiddleware\n"
        "class CustomHeaderMiddleware(BaseHTTPMiddleware):\n"
        "    async def dispatch(self, request, call_next):\n"
        "        response = await call_next(request)\n"
        "        response.headers['X-Custom-Header'] = 'from-custom-app'\n"
        "        return response\n"
        "app = FastAPI()\n"
        "app.add_middleware(CustomHeaderMiddleware)\n"
        "@app.get('/custom/mw')\n"
        "def mw(): return {'mw': True}\n",
        encoding="utf-8",
    )
    http_config = {"app": f"{f}:app"}

    app = FastAPI()

    @app.get("/builtin")
    def builtin():
        return {"builtin": True}

    with patch("agentseek_api.main.get_http_config", return_value=http_config), \
         patch("agentseek_api.main.get_config_dir", return_value=tmp_path):
        _merge_custom_app(app)

    client = TestClient(app)
    resp = client.get("/builtin")
    assert resp.status_code == 200
    assert resp.headers.get("X-Custom-Header") == "from-custom-app"


def test_merge_custom_app_no_lifespan_no_middleware(tmp_path: Path) -> None:
    """Custom app with no lifespan or middleware still works correctly."""
    f = _make_custom_app_file(tmp_path)
    http_config = {"app": f"{f}:app"}

    from contextlib import asynccontextmanager as acm
    from collections.abc import AsyncIterator

    @acm
    async def core_lifespan(a: FastAPI) -> AsyncIterator[None]:
        a.state.core_ran = True
        yield

    app = FastAPI(lifespan=core_lifespan)
    original_mw_count = len(app.user_middleware)

    with patch("agentseek_api.main.get_http_config", return_value=http_config), \
         patch("agentseek_api.main.get_config_dir", return_value=tmp_path):
        _merge_custom_app(app)

    assert len(app.user_middleware) == original_mw_count

    with TestClient(app) as client:
        assert getattr(app.state, "core_ran", False)
        resp = client.get("/custom/ping")
        assert resp.status_code == 200
