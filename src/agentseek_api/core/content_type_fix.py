"""Fix 422 errors when called from LangSmith Studio.

LangSmith Studio sends requests with Content-Type: text/plain even though the
body is valid JSON. FastAPI cannot parse the request body correctly in this case
and Pydantic validation fails with a 422 response.
This middleware patches the Content-Type to application/json at the ASGI layer.
"""

from __future__ import annotations

from starlette.types import ASGIApp, Receive, Scope, Send

_MUTATION_METHODS = (b"POST", b"PUT", b"PATCH")


def _is_plain_text(content_type: bytes) -> bool:
    """Check if content-type is text/plain, ignoring charset parameters."""
    normalized = content_type.lower().strip()
    if normalized == b"text/plain":
        return True
    if normalized.startswith(b"text/plain;"):
        return True
    return False


def _patch_content_type(headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]] | None:
    """Return a copy of headers with content-type fixed to application/json, or None if no fix needed."""
    for idx, (key, val) in enumerate(headers):
        if key == b"content-type" and _is_plain_text(val):
            patched = headers.copy()
            patched[idx] = (b"content-type", b"application/json")
            return patched
    return None


class ContentTypeFixMiddleware:
    """ASGI middleware that fixes text/plain Content-Type to avoid 422 errors."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        if isinstance(method, str):
            method = method.encode()

        if method not in _MUTATION_METHODS:
            await self.app(scope, receive, send)
            return

        patched = _patch_content_type(scope.get("headers", []))
        if patched is not None:
            scope["headers"] = patched

        await self.app(scope, receive, send)
