from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AssistantCreate(BaseModel):
    name: str
    graph_id: str = "default"


class AssistantRead(BaseModel):
    assistant_id: str
    name: str
    graph_id: str
    created_at: datetime


class ThreadCreate(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThreadRead(BaseModel):
    thread_id: str
    user_id: str
    metadata: dict[str, Any]
    created_at: datetime


class RunCreate(BaseModel):
    assistant_id: str
    input: dict[str, Any]


class RunResume(BaseModel):
    resume: Any


class RunRead(BaseModel):
    run_id: str
    thread_id: str
    assistant_id: str
    status: str
    output: dict[str, Any] | None
    interrupts: list[dict[str, Any]] | None = None
    last_error: str | None = None
