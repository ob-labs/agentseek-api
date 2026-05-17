from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProtocolCommandRequest(BaseModel):
    id: int
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class ProtocolEventStreamRequest(BaseModel):
    channels: list[str]
    namespaces: list[list[str]] | None = None
    depth: int | None = None
    since: int | None = None
