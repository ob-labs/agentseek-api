"""Rewrite text/plain Content-Type to application/json for mutation requests.

Some clients (notably LangGraph Studio) send valid JSON with
``Content-Type: text/plain``. FastAPI refuses to parse a text/plain body as
JSON and surfaces the raw string to Pydantic, producing
``model_attributes_type`` 422 errors on endpoints like ``/assistants/search``.

This is a zero-copy ASGI middleware — it only patches the scope headers,
never buffers or modifies the request body.
"""

from __future__ import annotations

from starlette.types import ASGIApp, Receive, Scope, Send

_TEXT_CONTENT_TYPES = (
    b"text/plain",
    b"text/plain;charset=utf-8",
    b"text/plain;charset=UTF-8",
    b"text/plain; charset=utf-8",
    b"text/plain; charset=UTF-8",
)

_METHODS_WITH_BODY = {b"POST", b"PUT", b"PATCH"}


class ContentTypeFixMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method_raw = scope.get("method", "")
        method: bytes = method_raw.encode() if isinstance(method_raw, str) else method_raw or b""
        if method not in _METHODS_WITH_BODY:
            await self.app(scope, receive, send)
            return

        headers: list[tuple[bytes, bytes]] = scope.get("headers", [])
        new_headers = None

        for i, (name, value) in enumerate(headers):
            if name == b"content-type" and value.lower() in _TEXT_CONTENT_TYPES:
                new_headers = list(headers)
                new_headers[i] = (b"content-type", b"application/json")
                break

        if new_headers is not None:
            scope["headers"] = new_headers

        await self.app(scope, receive, send)
