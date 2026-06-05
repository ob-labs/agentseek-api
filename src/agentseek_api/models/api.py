from datetime import datetime
from typing import Any
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AssistantCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_id: str | None = None
    graph_id: str
    config: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    if_exists: Literal["raise", "do_nothing"] = "raise"
    name: str = "Untitled"
    description: str | None = None


AssistantSortBy = Literal["assistant_id", "created_at", "updated_at", "name", "graph_id"]
AssistantSortOrder = Literal["asc", "desc"]
AssistantSelectField = Literal[
    "assistant_id", "graph_id", "name", "description",
    "config", "context", "created_at", "updated_at", "metadata", "version",
]


class AssistantCountRequest(BaseModel):
    metadata: dict[str, Any] | None = None
    graph_id: str | None = None
    name: str | None = None


class AssistantSearchRequest(BaseModel):
    metadata: dict[str, Any] | None = None
    graph_id: str | None = None
    name: str | None = None
    limit: int = Field(default=10, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    sort_by: AssistantSortBy | None = None
    sort_order: AssistantSortOrder | None = None
    select: list[AssistantSelectField] | None = None


class AssistantPatch(BaseModel):
    graph_id: str | None = None
    config: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    name: str | None = None
    description: str | None = None


class AssistantConfigRead(BaseModel):
    tags: list[str] = Field(default_factory=list)
    recursion_limit: int | None = None
    configurable: dict[str, Any] = Field(default_factory=dict)


class AssistantRead(BaseModel):
    assistant_id: str
    graph_id: str
    config: AssistantConfigRead
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]
    version: int = 1
    name: str = "Untitled"
    description: str | None = None


class AssistantVersionInfo(BaseModel):
    assistant_id: str
    current_version: int
    latest_version: int
    available_versions: list[int]
    supports_version_history: bool


class ErrorDetailResponse(BaseModel):
    detail: str


class Send(BaseModel):
    node: str
    input: Any


class Command(BaseModel):
    update: dict[str, Any] | list | None = None
    resume: Any = None
    goto: Send | list[Send] | str | list[str] | None = None


class ThreadSuperstepUpdate(BaseModel):
    values: list[dict[str, Any]] | dict[str, Any] | None = None
    command: Command | None = None
    as_node: str


class ThreadTTL(BaseModel):
    strategy: Literal["delete", "keep_latest"] = "delete"
    ttl: float


ThreadStatus = Literal["idle", "busy", "interrupted", "error"]


class ThreadSuperstep(BaseModel):
    updates: list[ThreadSuperstepUpdate]


class ThreadCreate(BaseModel):
    thread_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    if_exists: Literal["raise", "do_nothing"] = "raise"
    ttl: ThreadTTL | None = None
    supersteps: list[ThreadSuperstep] | None = None


ThreadSearchSortBy = Literal["thread_id", "status", "created_at", "updated_at", "state_updated_at"]
ThreadSearchSortOrder = Literal["asc", "desc"]
ThreadSearchSelectField = Literal[
    "thread_id", "created_at", "updated_at", "state_updated_at",
    "metadata", "config", "status", "values", "interrupts",
]


class ThreadSearchRequest(BaseModel):
    ids: list[str] | None = None
    metadata: dict[str, Any] | None = None
    values: dict[str, Any] | None = None
    status: ThreadStatus | None = None
    limit: int = Field(default=10, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    sort_by: ThreadSearchSortBy | None = None
    sort_order: ThreadSearchSortOrder | None = None
    select: list[ThreadSearchSelectField] | None = None
    extract: dict[str, str] | None = None


class ThreadPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, Any] | None = None


class ThreadCountRequest(BaseModel):
    metadata: dict[str, Any] | None = None
    values: dict[str, Any] | None = None
    status: ThreadStatus | None = None


class ThreadPruneRequest(BaseModel):
    thread_ids: list[str]
    strategy: Literal["delete", "keep_latest"] = "delete"


class ThreadPruneResponse(BaseModel):
    pruned_count: int


class ThreadTTLInfo(BaseModel):
    strategy: Literal["delete", "keep_latest"]
    ttl_minutes: float
    expires_at: datetime | None = None


class ThreadRead(BaseModel):
    thread_id: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]
    status: ThreadStatus = "idle"
    state_updated_at: datetime | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    values: dict[str, Any] = Field(default_factory=dict)
    interrupts: dict[str, Any] | None = None
    ttl: ThreadTTLInfo | None = None
    extracted: dict[str, Any] | None = None


class CheckpointConfig(BaseModel):
    thread_id: str | None = None
    checkpoint_ns: str | None = None
    checkpoint_id: str | None = None
    checkpoint_map: dict[str, Any] | None = None


class Interrupt(BaseModel):
    value: dict[str, Any]
    id: str | None = None


class ThreadStateTask(BaseModel):
    id: str
    name: str
    error: str | None = None
    interrupts: list[Interrupt] | None = None
    checkpoint: CheckpointConfig | None = None
    state: Any | None = None


class ThreadState(BaseModel):
    values: list[dict[str, Any]] | dict[str, Any]
    next: list[str]
    tasks: list[ThreadStateTask] = Field(default_factory=list)
    checkpoint: CheckpointConfig
    metadata: dict[str, Any]
    created_at: str | datetime
    parent_checkpoint: dict[str, Any] | None = None
    interrupts: list[Interrupt] | None = None


RunStreamMode = Literal[
    "values",
    "messages",
    "messages-tuple",
    "tasks",
    "checkpoints",
    "updates",
    "events",
    "debug",
    "custom",
]
RunInterrupt = Literal["*"] | list[str]
RunDurability = Literal["sync", "async", "exit"]
RunOnDisconnect = Literal["cancel", "continue"]
RunOnCompletion = Literal["delete", "keep"]
RunMultitaskStrategy = Literal["reject", "rollback", "interrupt", "enqueue"]
RunIfNotExists = Literal["create", "reject"]


class RunCreateStateful(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_id: str
    checkpoint: dict[str, Any] | None = None
    input: Any = None
    command: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    webhook: str | None = None
    interrupt_before: RunInterrupt | None = None
    interrupt_after: RunInterrupt | None = None
    stream_mode: RunStreamMode | list[RunStreamMode] | None = Field(default_factory=lambda: ["values"])
    stream_subgraphs: bool = False
    stream_resumable: bool = False
    feedback_keys: list[str] | None = None
    multitask_strategy: RunMultitaskStrategy = "enqueue"
    if_not_exists: RunIfNotExists = "reject"
    after_seconds: float | None = None
    checkpoint_during: bool = False
    durability: RunDurability = "async"


class RunCreateStreamingStateful(RunCreateStateful):
    on_disconnect: RunOnDisconnect = "continue"


class RunCreateStateless(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_id: str
    input: Any = None
    command: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    webhook: str | None = None
    stream_mode: RunStreamMode | list[RunStreamMode] | None = Field(default_factory=lambda: ["values"])
    feedback_keys: list[str] | None = None
    stream_subgraphs: bool = False
    stream_resumable: bool = False
    on_completion: RunOnCompletion = "keep"
    after_seconds: float | None = None
    checkpoint_during: bool = False
    durability: RunDurability = "async"


class RunCreateStreamingStateless(RunCreateStateless):
    on_disconnect: RunOnDisconnect = "continue"


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


class CronCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_id: str
    schedule: str
    timezone: str | None = None
    input: Any
    metadata: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    webhook: str | None = None
    enabled: bool = True


class CronSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_id: str | None = None
    enabled: bool | None = None
    thread_id: str | None = None
    limit: int = Field(default=10, ge=0)
    offset: int = Field(default=0, ge=0)


class CronCountRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_id: str | None = None
    enabled: bool | None = None
    thread_id: str | None = None


class CronPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schedule: str | None = None
    timezone: str | None = None
    input: Any | None = None
    metadata: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    webhook: str | None = None
    enabled: bool | None = None


class CronRead(BaseModel):
    cron_id: str
    assistant_id: str
    thread_id: str | None
    enabled: bool
    schedule: str
    timezone: str
    webhook: str | None = None
    next_run_at: datetime
    last_run_at: datetime | None = None
    last_tick_status: str | None = None
    last_error: str | None = None
    created_at: datetime
    updated_at: datetime | None = None


class CronSearchResponse(BaseModel):
    items: list[CronRead]


class CronCountResponse(BaseModel):
    count: int


class StorePutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace: list[str]
    key: str
    value: dict[str, Any]
    ttl: float | None = None


class StoreDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace: list[str]
    key: str


class StoreSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace_prefix: list[str] | None = None
    filter: dict[str, Any] | None = None
    query: str | None = None
    refresh_ttl: bool | None = None
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
