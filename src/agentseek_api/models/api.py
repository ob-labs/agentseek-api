from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AssistantCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    graph_id: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None


class AssistantSearchRequest(BaseModel):
    metadata: dict[str, Any] | None = None
    graph_id: str | None = None
    name: str | None = None
    limit: int = 10
    offset: int = 0
    sort_by: str = "created_at"
    sort_order: str = "desc"


class AssistantPatch(BaseModel):
    graph_id: str | None = None
    config: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    name: str | None = None
    description: str | None = None


class AssistantRead(BaseModel):
    assistant_id: str
    name: str
    graph_id: str
    created_at: datetime
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    description: str | None = None


class ThreadCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class ThreadSearchRequest(BaseModel):
    ids: list[str] | None = None
    metadata: dict[str, Any] | None = None
    status: str | None = None
    limit: int = 10
    offset: int = 0
    sort_by: str = "created_at"
    sort_order: str = "desc"


class ThreadPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, Any] | None = None


class ThreadPruneRequest(BaseModel):
    thread_ids: list[str]
    strategy: str = "keep_latest"


class ThreadRead(BaseModel):
    thread_id: str
    user_id: str
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime | None = None
    state_updated_at: datetime | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    status: str = "idle"


class RunCreate(BaseModel):
    assistant_id: str
    input: Any
    metadata: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    multitask_strategy: str = "enqueue"


class RunsCancelRequest(BaseModel):
    status: str | None = None
    thread_id: str | None = None
    run_ids: list[str] | None = None


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
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    multitask_strategy: str = "enqueue"


class StorePutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace: list[str]
    key: str
    value: dict[str, Any]


class StoreDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace: list[str]
    key: str


class StoreSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace_prefix: list[str] | None = None
    filter: dict[str, Any] | None = None
    limit: int = 10
    offset: int = 0


class StoreListNamespacesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prefix: list[str] | None = None
    suffix: list[str] | None = None
    max_depth: int | None = None
    limit: int = 100
    offset: int = 0


class StoreItemRead(BaseModel):
    namespace: list[str]
    key: str
    value: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class StoreSearchResponse(BaseModel):
    items: list[StoreItemRead]
