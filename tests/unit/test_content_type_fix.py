from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agentseek_api.core.content_type_fix import ContentTypeFixMiddleware


async def _echo_content_type(request: Request) -> JSONResponse:
    ct = request.headers.get("content-type", "")
    body = await request.body()
    return JSONResponse({"content_type": ct, "body": body.decode()})


app = Starlette(routes=[Route("/echo", _echo_content_type, methods=["POST"])])
app.add_middleware(ContentTypeFixMiddleware)
client = TestClient(app)


def test_text_plain_rewritten_to_json() -> None:
    resp = client.post("/echo", content='{"a":1}', headers={"content-type": "text/plain"})
    assert resp.status_code == 200
    assert resp.json()["content_type"] == "application/json"


def test_text_plain_with_charset_rewritten() -> None:
    resp = client.post("/echo", content='{}', headers={"content-type": "text/plain;charset=utf-8"})
    assert resp.status_code == 200
    assert resp.json()["content_type"] == "application/json"


def test_application_json_unchanged() -> None:
    resp = client.post("/echo", content='{}', headers={"content-type": "application/json"})
    assert resp.status_code == 200
    assert resp.json()["content_type"] == "application/json"
