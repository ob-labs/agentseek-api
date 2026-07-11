# Crons API Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the Crons API into alignment with the LangGraph Platform OpenAPI spec by adding all missing request/response fields and wiring their behavior, while keeping the project's existing working extensions (webhook delivery, timezone, last_* tracking, GET endpoint).

**Architecture:** Additive superset. New run-control fields are stored in the existing `kwargs_json` blob (mirroring `runs.py`) and applied at dispatch; `end_time` and `on_run_completed` become real `CronJob` columns enforced by the scheduler. Response gains spec fields plus a `next_run_at`→`next_run_date` rename (both exposed). `ThreadCronCreate` becomes a distinct schema.

**Tech Stack:** FastAPI, Pydantic v2, SQLAlchemy (async, `create_all` — no migrations), pytest/pytest-asyncio, in-memory SQLite for tests.

**Reference spec:** `docs/superpowers/specs/2026-06-13-crons-api-alignment-design.md`

---

## Conventions for every task

- Run tests with `uv run pytest <path> -v` (fall back to `pytest` if `uv` is unavailable).
- Commit after each task using the message in its final step. The suite must be green before committing.
- `docs/superpowers/` is **gitignored** — never `git add` the plan or spec. Only commit `src/` and `tests/` changes.
- The executor (`run_executor.execute_run`) reads only these keys from run kwargs: `config`, `context`, `command`, `stream_modes` (list), `interrupt_before`, `interrupt_after`, `durability`, `stream_subgraphs`. Unknown keys (e.g. `stream_resumable`, `multitask_strategy`) are ignored — safe to store.

## File Structure

| File | Responsibility |
|------|----------------|
| `src/agentseek_api/models/api.py` | Pydantic schemas: aliases, CronCreate, ThreadCronCreate, CronPatch, CronRead, CronSearchRequest, CronCountRequest |
| `src/agentseek_api/core/orm.py` | `CronJob` table: add `end_time`, `on_run_completed` |
| `src/agentseek_api/services/cron_service.py` | create/patch/search/count/read logic |
| `src/agentseek_api/services/cron_models.py` | `ClaimedCron` dataclass |
| `src/agentseek_api/services/cron_scheduler.py` | claim/dispatch/reconcile behavior |
| `src/agentseek_api/api/crons.py` | routes: ThreadCronCreate type, `response_model_exclude_none` |
| `tests/unit/test_cron_service.py`, `tests/integration/test_cron_api.py`, `tests/integration/test_scheduler_runtime.py` | tests |

## Task Overview (green-committable vertical slices)

- [ ] **Task 1** — ORM: add `end_time` + `on_run_completed` columns to `CronJob`
- [ ] **Task 2** — `CronRead` superset + `_to_read_model` + `response_model_exclude_none` on routes
- [ ] **Task 3** — `CronCreate` run-control + lifecycle fields; `_cron_kwargs` folding; `create_cron`
- [ ] **Task 4** — `ThreadCronCreate` separate schema + wire both thread-cron routes + `multitask_strategy`
- [ ] **Task 5** — `CronPatch` new fields
- [ ] **Task 6** — `CronSearchRequest`/`CronCountRequest`: metadata/sort/select + limit bounds
- [ ] **Task 7** — `ClaimedCron` carries `on_run_completed`/`end_time`/`multitask_strategy`
- [ ] **Task 8** — Scheduler: `end_time` enforcement in `claim_due_crons`
- [ ] **Task 9** — Scheduler: `multitask_strategy` passthrough in `dispatch_claimed_cron`
- [ ] **Task 10** — Scheduler: `on_run_completed` stateless-thread deletion in `_reconcile_terminal_ticks`
- [ ] **Task 11** — Full suite green + self-review

---

<!-- Task details appended below, one batch at a time -->

### Task 1 — ORM: add `end_time` + `on_run_completed` columns

**Files:**
- Modify: `src/agentseek_api/core/orm.py` (class `CronJob`, after line 76 `webhook` / before `max_webhook_attempts`)

Schema is created via `Base.metadata.create_all` (no migrations), and tests use a fresh in-memory SQLite DB, so adding columns is sufficient.

- [ ] **Step 1: Add the two columns**

In `src/agentseek_api/core/orm.py`, inside `class CronJob`, add these two lines immediately after the `webhook` column (line 76):

```python
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    on_run_completed: Mapped[str] = mapped_column(String(16), nullable=False, default="delete")
```

(`datetime`, `DateTime`, `String`, `Mapped`, `mapped_column` are already imported in this file.)

- [ ] **Step 2: Verify import + table build**

Run: `uv run python -c "from agentseek_api.core.orm import CronJob; print(CronJob.__table__.c.keys())"`
Expected: output list includes `end_time` and `on_run_completed`.

- [ ] **Step 3: Commit**

```bash
git add src/agentseek_api/core/orm.py
git commit -m "feat(crons): add end_time and on_run_completed columns to CronJob"
```

---

### Task 2 — `CronRead` superset + `_to_read_model` + `response_model_exclude_none`

Adds spec response fields (`user_id`, `payload`, `end_time`, `metadata`, `next_run_date`) while keeping `next_run_at` and extension fields. `_to_read_model` must be updated in the same task so the model always instantiates.

**Files:**
- Modify: `src/agentseek_api/models/api.py` (`CronRead`, lines 360-373)
- Modify: `src/agentseek_api/services/cron_service.py` (`_to_read_model`, lines 41-56)
- Modify: `src/agentseek_api/api/crons.py` (add `response_model_exclude_none=True` to the 5 routes returning `CronRead`)
- Test: `tests/unit/test_cron_service.py`

- [ ] **Step 1: Write failing test for the new read-model shape**

Append to `tests/unit/test_cron_service.py` (reuse the existing in-file pattern; this is a self-contained test):

```python
@pytest.mark.asyncio
async def test_to_read_model_exposes_spec_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "SEEKDB_URL", "sqlite+aiosqlite:///:memory:")

    class FakeCheckpointer:
        def __init__(self, connection_args: dict[str, str]) -> None:
            self.connection_args = connection_args

        def setup(self) -> None:
            return None

    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", FakeCheckpointer)

    manager = DatabaseManager()
    await manager.initialize()
    try:
        monkeypatch.setattr("agentseek_api.services.cron_service.db_manager", manager)
        user = User(identity="user-42", is_authenticated=True)

        created = await create_cron(
            assistant_id="assistant-1",
            thread_id=None,
            payload=CronCreate(
                assistant_id="assistant-1",
                schedule="FREQ=MINUTELY;INTERVAL=5",
                input={"kind": "read-model"},
                metadata={"source": "unit"},
                config={"model": "gpt-test"},
                context={"tenant": "acme"},
            ),
            user=user,
        )

        assert created.user_id == "user-42"
        assert created.metadata == {"source": "unit"}
        assert created.next_run_date == created.next_run_at
        assert created.end_time is None
        assert created.payload == {
            "input": {"kind": "read-model"},
            "config": {"model": "gpt-test"},
            "context": {"tenant": "acme"},
        }
    finally:
        await manager.close()
```

- [ ] **Step 2: Run it; expect failure**

Run: `uv run pytest tests/unit/test_cron_service.py::test_to_read_model_exposes_spec_fields -v`
Expected: FAIL — `CronRead` has no attribute `user_id` / `payload` / `next_run_date`.

- [ ] **Step 3: Replace `CronRead` in `models/api.py`**

Replace the whole `CronRead` class (lines 360-373) with:

```python
class CronRead(BaseModel):
    cron_id: str
    assistant_id: str
    thread_id: str | None
    user_id: str | None = None
    enabled: bool
    schedule: str
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    next_run_date: datetime
    next_run_at: datetime  # deprecated extension alias of next_run_date
    end_time: datetime | None = None
    created_at: datetime
    updated_at: datetime | None = None
    # Non-standard extensions (not in LangGraph spec):
    timezone: str
    webhook: str | None = None
    last_run_at: datetime | None = None
    last_tick_status: str | None = None
    last_error: str | None = None
```

- [ ] **Step 4: Replace `_to_read_model` in `cron_service.py`**

Replace the whole `_to_read_model` function (lines 41-56) with:

```python
def _to_read_model(row: CronJob) -> CronRead:
    kwargs = row.kwargs_json or {}
    payload = {
        "input": row.input_json,
        "config": kwargs.get("config", {}),
        "context": kwargs.get("context", {}),
    }
    return CronRead(
        cron_id=row.cron_id,
        assistant_id=row.assistant_id,
        thread_id=row.thread_id,
        user_id=row.user_id,
        enabled=row.enabled,
        schedule=row.schedule,
        payload=payload,
        metadata=row.metadata_json or {},
        next_run_date=row.next_run_at,
        next_run_at=row.next_run_at,
        end_time=row.end_time,
        created_at=row.created_at,
        updated_at=row.updated_at,
        timezone=row.timezone,
        webhook=row.webhook,
        last_run_at=row.last_run_at,
        last_tick_status=row.last_tick_status,
        last_error=row.last_error,
    )
```

- [ ] **Step 5: Add `response_model_exclude_none=True` to CronRead routes**

In `src/agentseek_api/api/crons.py`, change each of these 5 route decorators to add `response_model_exclude_none=True` (needed for Task 6 `select` projection and to suppress null extension fields):

```python
@router.post("/runs/crons", response_model=CronRead, response_model_exclude_none=True)
@router.post("/threads/{thread_id}/runs/crons", response_model=CronRead, response_model_exclude_none=True)
@router.get("/runs/crons/{cron_id}", response_model=CronRead, response_model_exclude_none=True)
@router.patch("/runs/crons/{cron_id}", response_model=CronRead, response_model_exclude_none=True)
```

(The search route returns `CronSearchResponse`; leave its decorator until Task 6.)

- [ ] **Step 6: Run the new test + full cron service tests**

Run: `uv run pytest tests/unit/test_cron_service.py -v`
Expected: PASS (all, including the new test).

- [ ] **Step 7: Commit**

```bash
git add src/agentseek_api/models/api.py src/agentseek_api/services/cron_service.py src/agentseek_api/api/crons.py tests/unit/test_cron_service.py
git commit -m "feat(crons): expand CronRead with spec fields (payload, user_id, metadata, end_time, next_run_date)"
```

> **Cross-task note (read before Task 3):** Tasks 3 and 10 change behavior that breaks
> *pre-existing* test assertions across multiple files. These are enumerated explicitly:
> - Task 3 introduces `stream_modes` into `kwargs_json`, invalidating four `kwargs_json ==`
>   assertions: `tests/integration/test_cron_api.py:69`, `tests/unit/test_cron_service.py:62`,
>   `tests/unit/test_cron_service.py:228`, `tests/integration/test_scheduler_runtime.py:180`.
>   Task 3 fixes all four.
> - Task 10 makes `on_run_completed="delete"` (the new default) actually delete stateless
>   threads, invalidating `tests/integration/test_scheduler_runtime.py:167-169` in
>   `test_dispatch_due_crons_creates_stateless_run_and_skips_busy_thread`. Task 10 fixes it.

---


### Task 3 — `CronCreate` run-control + lifecycle fields; `_cron_kwargs` folding; `create_cron`
**Files:**
- Modify: `src/agentseek_api/models/api.py:235` (add type aliases after `RunIfNotExists`), `src/agentseek_api/models/api.py:315-326` (`CronCreate`)
- Modify: `src/agentseek_api/services/cron_service.py:37-38` (`_cron_kwargs` + new helper), `src/agentseek_api/services/cron_service.py:70-92` (`create_cron`)
- Test: `tests/integration/test_cron_api.py`, `tests/unit/test_cron_service.py`

- [ ] **Step 1: Write failing test**
Append to `tests/integration/test_cron_api.py`:
```python
def test_create_stateless_cron_persists_run_control_and_lifecycle(client: TestClient) -> None:
    assistant_id = _create_assistant(client)

    response = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": {"kind": "run-control"},
            "config": {"model": "gpt-test"},
            "context": {"tenant": "acme"},
            "end_time": "2030-01-01T00:00:00+00:00",
            "on_run_completed": "keep",
            "interrupt_before": ["node_a"],
            "interrupt_after": "*",
            "stream_mode": ["values", "messages", "values"],
            "stream_subgraphs": True,
            "stream_resumable": True,
            "durability": "sync",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["end_time"] == "2030-01-01T00:00:00+00:00"
    assert body["on_run_completed"] == "keep"

    persisted = asyncio.run(_fetch_cron(body["cron_id"]))
    assert persisted is not None
    assert persisted.end_time is not None
    assert persisted.end_time.isoformat() == "2030-01-01T00:00:00+00:00"
    assert persisted.on_run_completed == "keep"
    assert persisted.kwargs_json == {
        "config": {"model": "gpt-test"},
        "context": {"tenant": "acme"},
        "stream_modes": ["values", "messages"],
        "interrupt_before": ["node_a"],
        "interrupt_after": "*",
        "durability": "sync",
        "stream_subgraphs": True,
        "stream_resumable": True,
    }


def test_create_stateless_cron_omits_default_run_control_from_kwargs(client: TestClient) -> None:
    assistant_id = _create_assistant(client)

    response = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": {"kind": "defaults"},
            "config": {"model": "gpt-test"},
            "context": {"tenant": "acme"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["end_time"] is None
    assert body["on_run_completed"] == "delete"

    persisted = asyncio.run(_fetch_cron(body["cron_id"]))
    assert persisted is not None
    assert persisted.end_time is None
    assert persisted.on_run_completed == "delete"
    assert persisted.kwargs_json == {
        "config": {"model": "gpt-test"},
        "context": {"tenant": "acme"},
        "stream_modes": ["values"],
    }
```
- [ ] **Step 2: Run it; expect failure**
Run: `uv run pytest tests/integration/test_cron_api.py::test_create_stateless_cron_persists_run_control_and_lifecycle tests/integration/test_cron_api.py::test_create_stateless_cron_omits_default_run_control_from_kwargs -v`
Expected: FAIL — `CronCreate` (`extra="forbid"`) rejects the new fields with 422, and `kwargs_json` lacks `stream_modes`/run-control keys.

- [ ] **Step 3: Add the locked type aliases**
In `src/agentseek_api/models/api.py`, immediately after the line `RunIfNotExists = Literal["create", "reject"]` (line 235), insert:
```python
CronOnRunCompleted = Literal["delete", "keep"]
CronSortBy = Literal["cron_id", "assistant_id", "thread_id", "next_run_date", "end_time", "created_at", "updated_at"]
CronSortOrder = Literal["asc", "desc"]
CronSelectField = Literal["cron_id", "assistant_id", "thread_id", "user_id", "enabled", "schedule", "payload", "metadata", "next_run_date", "next_run_at", "end_time", "created_at", "updated_at", "timezone", "webhook", "last_run_at", "last_tick_status", "last_error"]
```
- [ ] **Step 4: Expand `CronCreate`**
Replace the `CronCreate` class body (lines 315-326) with:
```python
class CronCreate(BaseModel):
    model_config = ConfigDict(extra="allow")

    assistant_id: str
    schedule: str
    timezone: str | None = None
    input: Any
    metadata: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    webhook: str | None = None
    enabled: bool = True
    end_time: datetime | None = None
    interrupt_before: RunInterrupt | None = None
    interrupt_after: RunInterrupt | None = None
    on_run_completed: CronOnRunCompleted = "delete"
    stream_mode: RunStreamMode | list[RunStreamMode] | None = Field(default_factory=lambda: ["values"])
    stream_subgraphs: bool = False
    stream_resumable: bool = False
    durability: RunDurability = "async"
```
- [ ] **Step 5: Add the stream-mode normalizer and fold run-control into `_cron_kwargs`**
In `src/agentseek_api/services/cron_service.py`, replace the existing `_cron_kwargs` definition (lines 37-38) with:
```python
def _cron_stream_modes(stream_mode: Any) -> list[str]:
    if stream_mode is None:
        return ["values"]
    raw = [stream_mode] if isinstance(stream_mode, str) else list(stream_mode)
    seen: set[str] = set()
    modes: list[str] = []
    for mode in raw:
        if mode not in seen:
            seen.add(mode)
            modes.append(mode)
    return modes


def _cron_kwargs(
    *,
    config: dict,
    context: dict,
    stream_mode: Any = None,
    interrupt_before: Any = None,
    interrupt_after: Any = None,
    durability: str = "async",
    stream_subgraphs: bool = False,
    stream_resumable: bool = False,
    multitask_strategy: str = "enqueue",
) -> dict:
    kwargs: dict[str, Any] = {"config": config, "context": context}
    kwargs["stream_modes"] = _cron_stream_modes(stream_mode)
    if interrupt_before is not None:
        kwargs["interrupt_before"] = interrupt_before
    if interrupt_after is not None:
        kwargs["interrupt_after"] = interrupt_after
    if durability != "async":
        kwargs["durability"] = durability
    if stream_subgraphs:
        kwargs["stream_subgraphs"] = True
    if stream_resumable:
        kwargs["stream_resumable"] = True
    if multitask_strategy != "enqueue":
        kwargs["multitask_strategy"] = multitask_strategy
    return kwargs
```
(The `multitask_strategy` param is added now so Task 4 only has to pass it at the call site. Default `stream_mode=None` makes `stream_modes` always present per the contract.)

- [ ] **Step 6: Update `create_cron` to persist lifecycle columns and run-control**
In `src/agentseek_api/services/cron_service.py`, replace the `CronJob(...)` construction inside `create_cron` (lines 76-88) with:
```python
        row = CronJob(
            assistant_id=assistant_id,
            thread_id=thread_id,
            user_id=user.identity,
            schedule=payload.schedule,
            timezone=timezone_name,
            enabled=payload.enabled,
            input_json=payload.input,
            metadata_json=payload.metadata,
            kwargs_json=_cron_kwargs(
                config=payload.config,
                context=payload.context,
                stream_mode=payload.stream_mode,
                interrupt_before=payload.interrupt_before,
                interrupt_after=payload.interrupt_after,
                durability=payload.durability,
                stream_subgraphs=payload.stream_subgraphs,
                stream_resumable=payload.stream_resumable,
                multitask_strategy=getattr(payload, "multitask_strategy", "enqueue"),
            ),
            webhook=webhook,
            end_time=payload.end_time,
            on_run_completed=getattr(payload, "on_run_completed", "delete"),
            next_run_at=compute_next_run_at(payload.schedule, timezone_name=timezone_name),
        )
```
(`getattr(...)` for `on_run_completed`/`multitask_strategy` lets the same body serve `ThreadCronCreate` in Task 4.)

- [ ] **Step 7: Run the new tests; expect pass**
Run: `uv run pytest tests/integration/test_cron_api.py::test_create_stateless_cron_persists_run_control_and_lifecycle tests/integration/test_cron_api.py::test_create_stateless_cron_omits_default_run_control_from_kwargs -v`
Expected: PASS — both new fields round-trip and the kwargs blob matches the contract exactly.

- [ ] **Step 8: Fix the four pre-existing `kwargs_json ==` assertions broken by `stream_modes`**
`stream_modes: ["values"]` is now always present. Update these existing assertions:

In `tests/integration/test_cron_api.py` line ~69 (`test_create_stateless_cron_persists_and_returns_resource`):
```python
    assert persisted.kwargs_json == {"config": {"model": "gpt-test"}, "context": {"tenant": "acme"}, "stream_modes": ["values"]}
```
In `tests/unit/test_cron_service.py` line ~62 (`test_create_cron_persists_webhook_timezone_and_runtime_kwargs`):
```python
        assert row.kwargs_json == {"config": {"model": "gpt-test"}, "context": {"tenant": "acme"}, "stream_modes": ["values"]}
```
In `tests/unit/test_cron_service.py` line ~228 (`test_patch_cron_updates_webhook_timezone_and_runtime_kwargs`): this asserts on a *patched* config/context; after Task 5 the patch re-folds run-control, so `stream_modes` will be present. Update to:
```python
        assert row.kwargs_json == {"config": {"temperature": 0.1}, "context": {"workspace": "west"}, "stream_modes": ["values"]}
```
Note: line 228's test passes only `config`/`context` to the patch; Task 5's re-fold logic preserves the original cron's `stream_modes` (which is `["values"]` since the create in that test sends no `stream_mode`). Verify after Task 5; if Task 5 is not yet applied, this specific assertion is fixed when Task 5 lands. Re-run after both Task 3 and Task 5.

In `tests/integration/test_scheduler_runtime.py` line ~180 (`test_dispatch_due_crons_creates_stateless_run_and_skips_busy_thread`): the dispatched `Run` inherits the cron's `kwargs_json`, which now carries `stream_modes`. Update to:
```python
    assert created_runs[0].kwargs_json == {"config": {"model": "gpt-test"}, "context": {"tenant": "acme"}, "stream_modes": ["values"]}
```
(This same test has a separate break from Task 10's `on_run_completed=delete` default — fixed in Task 10 Step 7. After Task 3 alone, the kwargs assertion is fixed but the thread-existence assertions at lines ~167-169 will still pass because deletion logic does not exist until Task 10.)

- [ ] **Step 9: Run the full cron + scheduler-runtime suites; expect pass**
Run: `uv run pytest tests/integration/test_cron_api.py tests/unit/test_cron_service.py tests/integration/test_scheduler_runtime.py -v`
Expected: PASS (the `test_cron_service.py` line-228 test is fully green only after Task 5; until then it may fail on the `stream_modes` re-fold — acceptable mid-sequence, confirmed in Task 5. All `test_scheduler_runtime.py` tests pass after the line-180 fix.)

- [ ] **Step 10: Commit**
```bash
git add src/agentseek_api/models/api.py src/agentseek_api/services/cron_service.py tests/integration/test_cron_api.py tests/unit/test_cron_service.py tests/integration/test_scheduler_runtime.py
git commit -m "feat(crons): add run-control and lifecycle fields to CronCreate"
```

---

### Task 4 — `ThreadCronCreate` separate schema + wire thread route + `multitask_strategy`
**Files:**
- Modify: `src/agentseek_api/models/api.py` (add `ThreadCronCreate` after `CronCreate`)
- Modify: `src/agentseek_api/api/crons.py:9-17` (import), `:35-45` (`_create_cron` param type), `:60-75` (`create_thread_cron`)
- Modify: `src/agentseek_api/services/cron_service.py` (widen `create_cron` param type)
- Test: `tests/integration/test_cron_api.py`

- [ ] **Step 1: Write failing test**
Append to `tests/integration/test_cron_api.py`:
```python
def test_create_thread_cron_persists_multitask_strategy(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client, user_id="owner")

    response = client.post(
        f"/threads/{thread_id}/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=HOURLY;INTERVAL=1",
            "input": {"kind": "thread-cron"},
            "config": {"model": "gpt-test"},
            "context": {"tenant": "acme"},
            "multitask_strategy": "rollback",
        },
        headers={"x-user-id": "owner"},
    )

    assert response.status_code == 200
    body = response.json()

    persisted = asyncio.run(_fetch_cron(body["cron_id"]))
    assert persisted is not None
    assert persisted.kwargs_json["multitask_strategy"] == "rollback"


def test_create_thread_cron_ignores_on_run_completed(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client, user_id="owner")

    # extra="allow" means on_run_completed is silently accepted (not 422) on the
    # thread-cron schema, which has no such field. It must NOT be persisted as a
    # column and must NOT leak into kwargs.
    response = client.post(
        f"/threads/{thread_id}/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=HOURLY;INTERVAL=1",
            "input": {"kind": "thread-cron"},
            "on_run_completed": "keep",
        },
        headers={"x-user-id": "owner"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["on_run_completed"] == "delete"

    persisted = asyncio.run(_fetch_cron(body["cron_id"]))
    assert persisted is not None
    assert persisted.on_run_completed == "delete"
    assert "on_run_completed" not in persisted.kwargs_json
```
(RESOLVED: because `ThreadCronCreate` uses `extra="allow"`, posting `on_run_completed` is silently accepted and ignored — NOT 422. The test asserts the column stays at default `"delete"` and the key never reaches `kwargs_json`.)

- [ ] **Step 2: Run it; expect failure**
Run: `uv run pytest tests/integration/test_cron_api.py::test_create_thread_cron_persists_multitask_strategy tests/integration/test_cron_api.py::test_create_thread_cron_ignores_on_run_completed -v`
Expected: FAIL — `multitask_strategy` is not folded into `kwargs_json` (CronCreate has no such field), and the thread route still uses `CronCreate`.

- [ ] **Step 3: Add `ThreadCronCreate`**
In `src/agentseek_api/models/api.py`, immediately after the `CronCreate` class (directly before `class CronSearchRequest`), insert:
```python
class ThreadCronCreate(BaseModel):
    model_config = ConfigDict(extra="allow")

    assistant_id: str
    schedule: str
    timezone: str | None = None
    input: Any
    metadata: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    webhook: str | None = None
    enabled: bool = True
    end_time: datetime | None = None
    interrupt_before: RunInterrupt | None = None
    interrupt_after: RunInterrupt | None = None
    multitask_strategy: RunMultitaskStrategy = "enqueue"
    stream_mode: RunStreamMode | list[RunStreamMode] | None = Field(default_factory=lambda: ["values"])
    stream_subgraphs: bool = False
    stream_resumable: bool = False
    durability: RunDurability = "async"
```
- [ ] **Step 4: Widen the service `create_cron` signature**
In `src/agentseek_api/services/cron_service.py`, add `ThreadCronCreate` to the imports from `agentseek_api.models.api` (alongside `CronCreate`), then change the `create_cron` signature (line 70) to:
```python
async def create_cron(*, assistant_id: str, thread_id: str | None, payload: CronCreate | ThreadCronCreate, user: User) -> CronRead:
```
(The `_cron_kwargs(... multitask_strategy=getattr(payload, "multitask_strategy", "enqueue"))` call and the `on_run_completed=getattr(...)` were already added in Task 3 Step 6, so the body needs no further change.)

- [ ] **Step 5: Point the thread route at `ThreadCronCreate`**
In `src/agentseek_api/api/crons.py`, add `ThreadCronCreate` to the imports from `agentseek_api.models.api`. Widen `_create_cron`'s param type (line 39) from `payload: CronCreate,` to `payload: CronCreate | ThreadCronCreate,`. Then change the `create_thread_cron` route signature (line 61) to:
```python
async def create_thread_cron(thread_id: str, payload: ThreadCronCreate, user: User = Depends(get_current_user)) -> CronRead:
```
- [ ] **Step 6: Run the new tests; expect pass**
Run: `uv run pytest tests/integration/test_cron_api.py::test_create_thread_cron_persists_multitask_strategy tests/integration/test_cron_api.py::test_create_thread_cron_ignores_on_run_completed -v`
Expected: PASS.

- [ ] **Step 7: Run the full cron suite; expect pass**
Run: `uv run pytest tests/integration/test_cron_api.py -v`
Expected: PASS — `test_create_thread_cron_persists_thread_and_user_binding` and the missing-thread test still pass against `ThreadCronCreate` (its fields are a superset of what those tests send).

- [ ] **Step 8: Commit**
```bash
git add src/agentseek_api/models/api.py src/agentseek_api/services/cron_service.py src/agentseek_api/api/crons.py tests/integration/test_cron_api.py
git commit -m "feat(crons): add ThreadCronCreate schema with multitask_strategy"
```

---

### Task 5 — `CronPatch` new fields + `patch_cron` folding
**Files:**
- Modify: `src/agentseek_api/models/api.py:347-357` (`CronPatch`)
- Modify: `src/agentseek_api/services/cron_service.py:106-159` (`patch_cron`)
- Test: `tests/integration/test_cron_api.py`

- [ ] **Step 1: Write failing test**
Append to `tests/integration/test_cron_api.py`:
```python
def test_patch_cron_updates_lifecycle_and_run_control(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": {"kind": "original"},
            "config": {"model": "gpt-test"},
            "context": {"tenant": "acme"},
        },
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    response = client.patch(
        f"/runs/crons/{cron_id}",
        json={
            "end_time": "2031-06-01T12:00:00+00:00",
            "durability": "sync",
            "stream_mode": ["updates", "values"],
            "on_run_completed": "keep",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["end_time"] == "2031-06-01T12:00:00+00:00"
    assert body["on_run_completed"] == "keep"

    persisted = asyncio.run(_fetch_cron(cron_id))
    assert persisted is not None
    assert persisted.end_time is not None
    assert persisted.end_time.isoformat() == "2031-06-01T12:00:00+00:00"
    assert persisted.on_run_completed == "keep"
    assert persisted.kwargs_json["config"] == {"model": "gpt-test"}
    assert persisted.kwargs_json["context"] == {"tenant": "acme"}
    assert persisted.kwargs_json["durability"] == "sync"
    assert persisted.kwargs_json["stream_modes"] == ["updates", "values"]


def test_patch_cron_config_only_preserves_run_control(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": {"kind": "original"},
            "config": {"model": "gpt-test"},
            "context": {"tenant": "acme"},
            "stream_mode": ["messages"],
            "durability": "sync",
        },
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    response = client.patch(
        f"/runs/crons/{cron_id}",
        json={"config": {"model": "gpt-next"}},
    )

    assert response.status_code == 200
    persisted = asyncio.run(_fetch_cron(cron_id))
    assert persisted is not None
    assert persisted.kwargs_json["config"] == {"model": "gpt-next"}
    assert persisted.kwargs_json["context"] == {"tenant": "acme"}
    assert persisted.kwargs_json["stream_modes"] == ["messages"]
    assert persisted.kwargs_json["durability"] == "sync"
```
- [ ] **Step 2: Run it; expect failure**
Run: `uv run pytest tests/integration/test_cron_api.py::test_patch_cron_updates_lifecycle_and_run_control tests/integration/test_cron_api.py::test_patch_cron_config_only_preserves_run_control -v`
Expected: FAIL — `CronPatch` (`extra="forbid"`) rejects the new fields with 422; `patch_cron` neither writes `end_time`/`on_run_completed` nor re-folds run-control, and clobbers run-control on a config-only patch.

- [ ] **Step 3: Expand `CronPatch`**
In `src/agentseek_api/models/api.py`, replace the `CronPatch` class body (lines 347-357) with:
```python
class CronPatch(BaseModel):
    model_config = ConfigDict(extra="allow")

    schedule: str | None = None
    timezone: str | None = None
    input: Any | None = None
    metadata: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    webhook: str | None = None
    enabled: bool | None = None
    end_time: datetime | None = None
    interrupt_before: RunInterrupt | None = None
    interrupt_after: RunInterrupt | None = None
    on_run_completed: CronOnRunCompleted | None = None
    stream_mode: RunStreamMode | list[RunStreamMode] | None = None
    stream_subgraphs: bool | None = None
    stream_resumable: bool | None = None
    durability: RunDurability | None = None
```
(Spec patch is `additionalProperties: true` → `extra="allow"`. `stream_mode` defaults to `None` so presence is detected via `model_fields_set`, not a default.)

- [ ] **Step 4: Update `patch_cron` to handle lifecycle columns and re-fold run-control**
In `src/agentseek_api/services/cron_service.py`, replace the `config`/`context` handling block (lines 140-153) with:
```python
        if "config" in payload.model_fields_set and payload.config is None:
            raise ValueError("config cannot be null")
        if "context" in payload.model_fields_set and payload.context is None:
            raise ValueError("context cannot be null")

        run_control_keys = {
            "config",
            "context",
            "stream_mode",
            "interrupt_before",
            "interrupt_after",
            "durability",
            "stream_subgraphs",
            "stream_resumable",
        }
        if run_control_keys & payload.model_fields_set:
            existing = row.kwargs_json or {}
            existing_stream = existing.get("stream_modes")
            row.kwargs_json = _cron_kwargs(
                config=payload.config if "config" in payload.model_fields_set else existing.get("config", {}),
                context=payload.context if "context" in payload.model_fields_set else existing.get("context", {}),
                stream_mode=payload.stream_mode if "stream_mode" in payload.model_fields_set else existing_stream,
                interrupt_before=payload.interrupt_before if "interrupt_before" in payload.model_fields_set else existing.get("interrupt_before"),
                interrupt_after=payload.interrupt_after if "interrupt_after" in payload.model_fields_set else existing.get("interrupt_after"),
                durability=payload.durability if "durability" in payload.model_fields_set else existing.get("durability", "async"),
                stream_subgraphs=payload.stream_subgraphs if "stream_subgraphs" in payload.model_fields_set else existing.get("stream_subgraphs", False),
                stream_resumable=payload.stream_resumable if "stream_resumable" in payload.model_fields_set else existing.get("stream_resumable", False),
                multitask_strategy=existing.get("multitask_strategy", "enqueue"),
            )

        if "end_time" in payload.model_fields_set:
            row.end_time = payload.end_time
        if "on_run_completed" in payload.model_fields_set:
            if payload.on_run_completed is None:
                raise ValueError("on_run_completed cannot be null")
            row.on_run_completed = payload.on_run_completed
```
(Re-folding from `existing` preserves a previously stored `multitask_strategy` and only overwrites keys present in the patch. `_cron_stream_modes` re-dedupes the already-normalized `existing_stream` idempotently, so config-only patches keep the prior `stream_modes`.)

- [ ] **Step 5: Run the new tests; expect pass**
Run: `uv run pytest tests/integration/test_cron_api.py::test_patch_cron_updates_lifecycle_and_run_control tests/integration/test_cron_api.py::test_patch_cron_config_only_preserves_run_control -v`
Expected: PASS.

- [ ] **Step 6: Run the full cron suites; expect pass (closes Task 3 Step 8's line-228 dependency)**
Run: `uv run pytest tests/integration/test_cron_api.py tests/unit/test_cron_service.py -v`
Expected: PASS — including `test_patch_cron_updates_webhook_timezone_and_runtime_kwargs` (line 228), now that the re-fold adds `stream_modes: ["values"]` (matching the Task 3 Step 8 assertion update), and `test_patch_cron_rejects_explicit_null_input` (null guard unchanged).

- [ ] **Step 7: Commit**
```bash
git add src/agentseek_api/models/api.py src/agentseek_api/services/cron_service.py tests/integration/test_cron_api.py
git commit -m "feat(crons): support lifecycle and run-control fields in CronPatch"
```

---

### Task 6 — Search/Count: metadata filter, sort, select projection + limit bounds
**Files:**
- Modify: `src/agentseek_api/models/api.py:329-345` (`CronSearchRequest`, `CronCountRequest`)
- Modify: `src/agentseek_api/services/cron_service.py:59-67` (`_search_stmt`), `:174-186` (`search_crons`), `:189-196` (`count_crons`)
- Modify: `src/agentseek_api/api/crons.py:78-87` (search route)
- Test: `tests/integration/test_cron_api.py`

**Select projection — RESOLVED approach:** Mirror `search_assistants` (`api/assistants.py:141-163`): the service `search_crons` always builds full `CronRead` items. The route keeps `response_model=CronSearchResponse` for the default case, but when `payload.select` is set it returns a `JSONResponse` whose body is `{"items": [item.model_dump(mode="json", include=fields) for item in result.items]}`, bypassing the response model. `CronSearchResponse.items: list[CronRead]` stays unchanged.

- [ ] **Step 1: Write failing test**
Append to `tests/integration/test_cron_api.py`:
```python
def test_search_crons_sorts_by_next_run_date_asc(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    hourly = client.post(
        "/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=HOURLY;INTERVAL=1", "input": {"k": "hourly"}},
    ).json()
    minutely = client.post(
        "/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=MINUTELY;INTERVAL=1", "input": {"k": "minutely"}},
    ).json()

    response = client.post(
        "/runs/crons/search",
        json={"assistant_id": assistant_id, "sort_by": "next_run_date", "sort_order": "asc"},
    )
    assert response.status_code == 200
    ids = [item["cron_id"] for item in response.json()["items"]]
    assert ids == [minutely["cron_id"], hourly["cron_id"]]


def test_search_crons_filters_by_metadata(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    matching = client.post(
        "/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=MINUTELY;INTERVAL=5", "input": {"k": 1}, "metadata": {"team": "alpha"}},
    ).json()
    client.post(
        "/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=MINUTELY;INTERVAL=5", "input": {"k": 2}, "metadata": {"team": "beta"}},
    )

    response = client.post(
        "/runs/crons/search",
        json={"assistant_id": assistant_id, "metadata": {"team": "alpha"}},
    )
    assert response.status_code == 200
    ids = [item["cron_id"] for item in response.json()["items"]]
    assert ids == [matching["cron_id"]]

    count_response = client.post(
        "/runs/crons/count",
        json={"assistant_id": assistant_id, "metadata": {"team": "alpha"}},
    )
    assert count_response.status_code == 200
    assert count_response.json() == {"count": 1}


def test_search_crons_select_returns_subset(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=MINUTELY;INTERVAL=5", "input": {"k": 1}},
    ).json()

    response = client.post(
        "/runs/crons/search",
        json={"assistant_id": assistant_id, "select": ["cron_id", "schedule"]},
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert items == [{"cron_id": created["cron_id"], "schedule": "FREQ=MINUTELY;INTERVAL=5"}]


def test_search_crons_limit_bounds(client: TestClient) -> None:
    assistant_id = _create_assistant(client)

    zero = client.post("/runs/crons/search", json={"assistant_id": assistant_id, "limit": 0})
    assert zero.status_code == 422

    one = client.post("/runs/crons/search", json={"assistant_id": assistant_id, "limit": 1})
    assert one.status_code == 200
```
- [ ] **Step 2: Run it; expect failure**
Run: `uv run pytest tests/integration/test_cron_api.py -k "sorts_by_next_run_date or filters_by_metadata or select_returns_subset or limit_bounds" -v`
Expected: FAIL — `CronSearchRequest`/`CronCountRequest` reject `metadata`/`sort_by`/`sort_order`/`select` with 422; `limit: 0` is currently accepted (200, not 422); sort defaults to `created_at desc`.

- [ ] **Step 3: Expand `CronSearchRequest` and `CronCountRequest`**
In `src/agentseek_api/models/api.py`, replace the `CronSearchRequest` body (lines 329-336) with:
```python
class CronSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_id: str | None = None
    enabled: bool | None = None
    thread_id: str | None = None
    metadata: dict[str, Any] | None = None
    limit: int = Field(default=10, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    sort_by: CronSortBy | None = None
    sort_order: CronSortOrder | None = None
    select: list[CronSelectField] | None = None
```
and replace the `CronCountRequest` body (lines 339-345) with:
```python
class CronCountRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_id: str | None = None
    enabled: bool | None = None
    thread_id: str | None = None
    metadata: dict[str, Any] | None = None
```
(`extra="forbid"` retained — spec search/count bodies are `additionalProperties: false`; `enabled`/`thread_id` kept as declared extension fields.)

- [ ] **Step 4: Add metadata filtering to `_search_stmt`**
In `src/agentseek_api/services/cron_service.py`, replace `_search_stmt` (lines 59-67) with:
```python
def _search_stmt(*, payload: CronSearchRequest | CronCountRequest):
    stmt = select(CronJob)
    if payload.assistant_id is not None:
        stmt = stmt.where(CronJob.assistant_id == payload.assistant_id)
    if payload.enabled is not None:
        stmt = stmt.where(CronJob.enabled == payload.enabled)
    if payload.thread_id is not None:
        stmt = stmt.where(CronJob.thread_id == payload.thread_id)
    if payload.metadata is not None:
        for key, value in payload.metadata.items():
            stmt = stmt.where(CronJob.metadata_json[key].as_string() == str(value))
    return stmt
```
- [ ] **Step 5: Apply sort to `search_crons`**
In `src/agentseek_api/services/cron_service.py`, replace the body of `search_crons` (lines 174-186) with:
```python
async def search_crons(*, payload: CronSearchRequest, user: User, filters: dict[str, Any] | None = None) -> CronSearchResponse:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = _search_stmt(payload=payload)
        stmt = apply_metadata_filters(stmt, CronJob, filters)

        sort_columns = {
            "next_run_date": CronJob.next_run_at,
            "end_time": CronJob.end_time,
        }
        if payload.sort_by is not None:
            sort_column = sort_columns.get(payload.sort_by, getattr(CronJob, payload.sort_by, CronJob.created_at))
            order = sort_column.asc() if payload.sort_order == "asc" else sort_column.desc()
            stmt = stmt.order_by(order, CronJob.cron_id.desc())
        else:
            stmt = stmt.order_by(CronJob.created_at.desc(), CronJob.cron_id.desc())

        stmt = stmt.limit(payload.limit).offset(payload.offset)
        rows = list((await session.scalars(stmt)).all())
    return CronSearchResponse(items=[_to_read_model(row) for row in rows])
```
(`next_run_date`→`next_run_at` column, `end_time`→`end_time`; other `sort_by` values resolve via `getattr(CronJob, ...)`. Default ordering unchanged.)

- [ ] **Step 6: Confirm `count_crons` picks up metadata filtering**
The `count_crons` body (lines 189-196) already calls `_search_stmt(payload=payload)`, which now includes the metadata filter — no further change needed. Confirm it reads:
```python
async def count_crons(*, payload: CronCountRequest, user: User, filters: dict[str, Any] | None = None) -> CronCountResponse:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        stmt = _search_stmt(payload=payload)
        stmt = apply_metadata_filters(stmt, CronJob, filters)
        stmt = stmt.with_only_columns(func.count(CronJob.cron_id))
        count = await session.scalar(stmt)
    return CronCountResponse(count=int(count or 0))
```
- [ ] **Step 7: Implement `select` projection in the search route**
In `src/agentseek_api/api/crons.py`, change the existing `from fastapi.responses import Response` import to `from fastapi.responses import JSONResponse, Response`. Then replace the `search_crons` route (lines 78-81) with:
```python
@router.post("/runs/crons/search", response_model=CronSearchResponse)
async def search_crons(payload: CronSearchRequest, user: User = Depends(get_current_user)) -> CronSearchResponse | JSONResponse:
    filters = await authorize(user, "crons", "search", {})
    result = await cron_service.search_crons(payload=payload, user=user, filters=filters)
    if payload.select is not None:
        fields = set(payload.select)
        return JSONResponse(
            content={"items": [item.model_dump(mode="json", include=fields) for item in result.items]}
        )
    return result
```
(RESOLVED: mirrors `search_assistants`. `CronSelectField` includes spec names `payload`/`next_run_date` which are not `CronRead` attributes, so `model_dump(include=...)` silently drops them — acceptable for an additive superset; round-trippable fields project correctly.)

- [ ] **Step 8: Run the new tests; expect pass**
Run: `uv run pytest tests/integration/test_cron_api.py -k "sorts_by_next_run_date or filters_by_metadata or select_returns_subset or limit_bounds" -v`
Expected: PASS.

- [ ] **Step 9: Run the full cron suite; expect pass**
Run: `uv run pytest tests/integration/test_cron_api.py -v`
Expected: PASS — `test_search_crons_rejects_negative_limit_and_offset` (limit=-1) still 422s under `ge=1`; `test_search_count_get_patch_and_delete_crons` still passes (default ordering/no-select path unchanged).

- [ ] **Step 10: Commit**
```bash
git add src/agentseek_api/models/api.py src/agentseek_api/services/cron_service.py src/agentseek_api/api/crons.py tests/integration/test_cron_api.py
git commit -m "feat(crons): add metadata filter, sort, and select to search and count"
```

---

### Task 7 — ClaimedCron carries on_run_completed/end_time/multitask_strategy
**Files:**
- Modify: `src/agentseek_api/services/cron_models.py`
- Test: `tests/unit/test_scheduler.py`

- [ ] **Step 1: Write failing test** — append to `tests/unit/test_scheduler.py` (it builds real DB-backed `CronJob` rows; mirror that exact setup using the existing module-level `_FakeCheckpointer` / `_as_utc` and the `Assistant`/`CronJob` imports already at the top of the file):
```python
@pytest.mark.asyncio
async def test_claim_due_crons_maps_run_control_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from agentseek_api.services.cron_scheduler import claim_due_crons
    from agentseek_api.settings import settings

    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/scheduler-runctl.db")
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", _FakeCheckpointer)
    await db_manager.close()
    await db_manager.initialize()
    try:
        session_factory = db_manager.get_session_factory()
        due_at = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=1)
        end_at = due_at + timedelta(days=1)
        async with session_factory() as session:
            assistant = Assistant(name="scheduler-runctl", graph_id="default")
            session.add(assistant)
            await session.flush()
            cron = CronJob(
                assistant_id=assistant.assistant_id,
                thread_id=None,
                user_id="u1",
                schedule="FREQ=MINUTELY;INTERVAL=1",
                enabled=True,
                input_json={"kind": "unit"},
                next_run_at=due_at,
                end_time=end_at,
                on_run_completed="keep",
                kwargs_json={"config": {}, "context": {}, "multitask_strategy": "interrupt"},
            )
            session.add(cron)
            await session.commit()
            cron_id = cron.cron_id

        claimed = await claim_due_crons(limit=10, scheduler_id="scheduler-a", now=due_at)

        assert [item.cron_id for item in claimed] == [cron_id]
        item = claimed[0]
        assert item.on_run_completed == "keep"
        assert _as_utc(item.end_time) == end_at
        assert item.multitask_strategy == "interrupt"
    finally:
        await db_manager.close()


@pytest.mark.asyncio
async def test_claim_due_crons_run_control_fields_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from agentseek_api.services.cron_scheduler import claim_due_crons
    from agentseek_api.settings import settings

    monkeypatch.setattr(settings, "SEEKDB_URL", f"sqlite+aiosqlite:///{tmp_path}/scheduler-runctl-default.db")
    monkeypatch.setattr("agentseek_api.core.database.OceanBaseCheckpointSaver", _FakeCheckpointer)
    await db_manager.close()
    await db_manager.initialize()
    try:
        session_factory = db_manager.get_session_factory()
        due_at = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=1)
        async with session_factory() as session:
            assistant = Assistant(name="scheduler-runctl-default", graph_id="default")
            session.add(assistant)
            await session.flush()
            cron = CronJob(
                assistant_id=assistant.assistant_id,
                thread_id=None,
                user_id="u1",
                schedule="FREQ=MINUTELY;INTERVAL=1",
                enabled=True,
                input_json={"kind": "unit"},
                next_run_at=due_at,
            )
            session.add(cron)
            await session.commit()

        claimed = await claim_due_crons(limit=10, scheduler_id="scheduler-a", now=due_at)

        assert len(claimed) == 1
        item = claimed[0]
        assert item.on_run_completed == "delete"
        assert item.end_time is None
        assert item.multitask_strategy == "enqueue"
    finally:
        await db_manager.close()
```
- [ ] **Step 2: Run it; expect failure**
Run: `uv run pytest tests/unit/test_scheduler.py::test_claim_due_crons_maps_run_control_fields tests/unit/test_scheduler.py::test_claim_due_crons_run_control_fields_default -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument` / `AttributeError: 'ClaimedCron' object has no attribute 'on_run_completed'`.

- [ ] **Step 3: Add the three fields and populate them in `from_row`** — replace the `ClaimedCron` dataclass in `src/agentseek_api/services/cron_models.py` with (new fields at the END, each defaulted so frozen+slots field ordering stays valid):
```python
@dataclass(frozen=True, slots=True)
class ClaimedCron:
    tick_id: int
    cron_id: str
    assistant_id: str
    thread_id: str | None
    run_id: str | None
    user_id: str
    schedule: str
    input_json: Any
    metadata_json: dict[str, Any]
    kwargs_json: dict[str, Any]
    scheduled_for: datetime
    on_run_completed: str = "delete"
    end_time: datetime | None = None
    multitask_strategy: str = "enqueue"

    @classmethod
    def from_row(
        cls,
        row: CronJob,
        *,
        tick_id: int,
        scheduled_for: datetime,
        thread_id: str | None = None,
        run_id: str | None = None,
    ) -> "ClaimedCron":
        return cls(
            tick_id=tick_id,
            cron_id=row.cron_id,
            assistant_id=row.assistant_id,
            thread_id=thread_id if thread_id is not None else row.thread_id,
            run_id=run_id,
            user_id=row.user_id,
            schedule=row.schedule,
            input_json=row.input_json,
            metadata_json=row.metadata_json,
            kwargs_json=row.kwargs_json,
            scheduled_for=scheduled_for,
            on_run_completed=row.on_run_completed,
            end_time=row.end_time,
            multitask_strategy=(row.kwargs_json or {}).get("multitask_strategy", "enqueue"),
        )
```
- [ ] **Step 4: Run it; expect pass**
Run: `uv run pytest tests/unit/test_scheduler.py::test_claim_due_crons_maps_run_control_fields tests/unit/test_scheduler.py::test_claim_due_crons_run_control_fields_default -v`
Expected: PASS.

- [ ] **Step 5: Run the full unit scheduler suite to confirm no regressions**
Run: `uv run pytest tests/unit/test_scheduler.py -v`
Expected: PASS — all green.

- [ ] **Step 6: Commit**
```bash
git add src/agentseek_api/services/cron_models.py tests/unit/test_scheduler.py
git commit -m "feat(crons): carry on_run_completed/end_time/multitask_strategy on ClaimedCron"
```

---

### Task 8 — end_time enforcement in claim_due_crons
**Files:**
- Modify: `src/agentseek_api/services/cron_scheduler.py`
- Test: `tests/integration/test_scheduler_runtime.py`

- [ ] **Step 1: Write failing test** — first add this helper near the other helpers at the top of `tests/integration/test_scheduler_runtime.py` (mirrors the existing `_set_cron_webhook`):
```python
async def _set_cron_end_time(cron_id: str, *, end_time: datetime | None) -> None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        cron = await session.scalar(select(CronJob).where(CronJob.cron_id == cron_id))
        assert cron is not None
        cron.end_time = end_time
        await session.commit()
```
then append the two tests:
```python
def test_dispatch_due_crons_skips_cron_past_end_time(client: TestClient) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=MINUTELY;INTERVAL=1", "input": {"kind": "past-end-time"}},
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    due_at = datetime.now(UTC) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(cron_id, when=due_at))
    asyncio.run(_set_cron_end_time(cron_id, end_time=due_at - timedelta(minutes=5)))

    results = asyncio.run(cron_scheduler_module.dispatch_due_crons(limit=10, scheduler_id="scheduler-1", now=due_at))

    assert results == []
    ticks = asyncio.run(_list_ticks_for_cron(cron_id))
    assert ticks == []


def test_dispatch_due_crons_disables_cron_when_next_run_crosses_end_time(client: TestClient) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=MINUTELY;INTERVAL=1", "input": {"kind": "cross-end-time"}},
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    due_at = datetime.now(UTC) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(cron_id, when=due_at))
    # end_time is after this fire (so it fires now) but before the next computed run_at.
    asyncio.run(_set_cron_end_time(cron_id, end_time=due_at + timedelta(seconds=30)))

    results = asyncio.run(cron_scheduler_module.dispatch_due_crons(limit=10, scheduler_id="scheduler-1", now=due_at))

    assert len(results) == 1
    assert results[0].status == "queued"

    persisted = asyncio.run(_fetch_cron(cron_id))
    assert persisted is not None
    assert persisted.enabled is False
    assert _as_utc(persisted.next_run_at) > _as_utc(persisted.end_time)
```
Note: the `delete` test's stateless cron defaults to `on_run_completed="delete"`, but it never fires (skipped past end_time), so no thread is created — no interaction with Task 10. The `cross-end-time` test DOES fire; its stateless thread will be deleted by Task 10's logic once that lands, but these asserts only check `results`/`enabled`/`next_run_at`, so they remain valid regardless.

- [ ] **Step 2: Run it; expect failure**
Run: `uv run pytest tests/integration/test_scheduler_runtime.py::test_dispatch_due_crons_skips_cron_past_end_time tests/integration/test_scheduler_runtime.py::test_dispatch_due_crons_disables_cron_when_next_run_crosses_end_time -v`
Expected: FAIL — the past-end-time cron is still claimed (returns a result + creates a tick), and the crossing cron stays `enabled=True`.

- [ ] **Step 3: Add the end_time WHERE filter to the due-cron query** in `claim_due_crons` — edit the due-cron `select(CronJob)` block:
```python
            rows = list(
                (
                    await session.scalars(
                        select(CronJob)
                        .where(
                            CronJob.enabled.is_(True),
                            CronJob.next_run_at <= current_time,
                            ((CronJob.end_time.is_(None)) | (CronJob.end_time > current_time)),
                        )
                        .order_by(CronJob.next_run_at.asc(), CronJob.cron_id.asc())
                        .limit(remaining)
                        .with_for_update()
                    )
                ).all()
            )
```
- [ ] **Step 4: Disable the cron once the newly computed next_run_at passes end_time** — in the same `for row in rows:` loop, edit the try/except that advances `next_run_at`:
```python
                claimed.append(ClaimedCron.from_row(row, tick_id=tick.id, scheduled_for=scheduled_for))
                try:
                    row.next_run_at = compute_next_run_at(row.schedule, timezone_name=row.timezone, now=current_time)
                except ValueError:
                    row.enabled = False
                else:
                    if row.end_time is not None and row.next_run_at > row.end_time:
                        row.enabled = False
```
- [ ] **Step 5: Run it; expect pass**
Run: `uv run pytest tests/integration/test_scheduler_runtime.py::test_dispatch_due_crons_skips_cron_past_end_time tests/integration/test_scheduler_runtime.py::test_dispatch_due_crons_disables_cron_when_next_run_crosses_end_time -v`
Expected: PASS.

- [ ] **Step 6: Run the full integration scheduler runtime suite**
Run: `uv run pytest tests/integration/test_scheduler_runtime.py -v`
Expected: PASS — all green.

- [ ] **Step 7: Commit**
```bash
git add src/agentseek_api/services/cron_scheduler.py tests/integration/test_scheduler_runtime.py
git commit -m "feat(crons): enforce end_time in claim_due_crons and disable expired crons"
```

---

### Task 9 — multitask_strategy passthrough in dispatch_claimed_cron
**Files:**
- Modify: `src/agentseek_api/services/cron_scheduler.py`
- Test: `tests/integration/test_scheduler_runtime.py`

- [ ] **Step 1: Write failing test** — append to `tests/integration/test_scheduler_runtime.py`. Creates a thread cron with `multitask_strategy` (enabled by Task 4), fires it, asserts the created `Run` row carries the value via `_list_runs_for_thread`:
```python
def test_dispatch_due_crons_passes_multitask_strategy_to_thread_run(client: TestClient) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client, user_id="owner")

    created = client.post(
        f"/threads/{thread_id}/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=1",
            "input": {"kind": "thread-bound"},
            "multitask_strategy": "interrupt",
        },
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200

    due_at = datetime.now(UTC) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(created.json()["cron_id"], when=due_at))

    results = asyncio.run(cron_scheduler_module.dispatch_due_crons(limit=10, scheduler_id="scheduler-1", now=due_at))

    assert len(results) == 1
    assert results[0].status == "queued"
    assert results[0].thread_id == thread_id

    runs = asyncio.run(_list_runs_for_thread(thread_id))
    assert len(runs) == 1
    assert runs[0].multitask_strategy == "interrupt"
```
- [ ] **Step 2: Run it; expect failure**
Run: `uv run pytest tests/integration/test_scheduler_runtime.py::test_dispatch_due_crons_passes_multitask_strategy_to_thread_run -v`
Expected: FAIL — `prepare_run` is called without `multitask_strategy`, so the `Run` defaults to `"enqueue"`.

- [ ] **Step 3: Pass `multitask_strategy` into both `prepare_run` calls in `dispatch_claimed_cron`** — edit `src/agentseek_api/services/cron_scheduler.py`. Stateless branch (after creating the ephemeral thread):
```python
            run, _graph_id = await prepare_run(
                thread_id=thread.thread_id,
                assistant_id=claim.assistant_id,
                payload=claim.input_json,
                user=user,
                metadata={
                    **claim.metadata_json,
                    "cron_id": claim.cron_id,
                    "scheduled_for": scheduled_for_iso,
                },
                kwargs=claim.kwargs_json,
                multitask_strategy=claim.multitask_strategy,
                tick_id=claim.tick_id,
            )
```
Thread-cron branch:
```python
        run, _graph_id = await prepare_run(
            thread_id=claim.thread_id,
            assistant_id=claim.assistant_id,
            payload=claim.input_json,
            user=user,
            metadata={
                **claim.metadata_json,
                "cron_id": claim.cron_id,
                "scheduled_for": scheduled_for_iso,
            },
            kwargs=claim.kwargs_json,
            multitask_strategy=claim.multitask_strategy,
            tick_id=claim.tick_id,
        )
```
- [ ] **Step 4: Run it; expect pass**
Run: `uv run pytest tests/integration/test_scheduler_runtime.py::test_dispatch_due_crons_passes_multitask_strategy_to_thread_run -v`
Expected: PASS.

- [ ] **Step 5: Run the full integration scheduler runtime suite**
Run: `uv run pytest tests/integration/test_scheduler_runtime.py -v`
Expected: PASS — all green.

- [ ] **Step 6: Commit**
```bash
git add src/agentseek_api/services/cron_scheduler.py tests/integration/test_scheduler_runtime.py
git commit -m "feat(crons): pass multitask_strategy through to cron-dispatched runs"
```

---

### Task 10 — on_run_completed stateless-thread deletion in _reconcile_terminal_ticks
**Files:**
- Modify: `src/agentseek_api/services/cron_scheduler.py`
- Modify: `tests/integration/test_scheduler_runtime.py` (existing stateless test + new tests)
- Test: `tests/integration/test_scheduler_runtime.py`

- [ ] **Step 1: Write failing tests** — append to `tests/integration/test_scheduler_runtime.py`. Add a single-thread fetch helper (mirrors existing helper style), then three tests for delete / keep / caller-owned. Stateless threads are identified by `metadata_json["cron_id"]` as the existing stateless test does:
```python
async def _fetch_thread(thread_id: str) -> Thread | None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        return await session.scalar(select(Thread).where(Thread.thread_id == thread_id))


def test_dispatch_due_crons_deletes_stateless_thread_on_run_completed_delete(client: TestClient) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=1",
            "input": {"kind": "stateless-delete"},
            "on_run_completed": "delete",
        },
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    due_at = datetime.now(UTC) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(cron_id, when=due_at))

    results = asyncio.run(cron_scheduler_module.dispatch_due_crons(limit=10, scheduler_id="scheduler-1", now=due_at))
    assert len(results) == 1
    assert results[0].status == "queued"

    user_threads = asyncio.run(_list_threads_for_user("owner"))
    stateless_threads = [t for t in user_threads if t.metadata_json.get("cron_id") == cron_id]
    assert stateless_threads == []

    ticks = asyncio.run(_list_ticks_for_cron(cron_id))
    assert len(ticks) == 1
    assert ticks[0].status == "success"


def test_dispatch_due_crons_keeps_stateless_thread_on_run_completed_keep(client: TestClient) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=1",
            "input": {"kind": "stateless-keep"},
            "on_run_completed": "keep",
        },
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    due_at = datetime.now(UTC) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(cron_id, when=due_at))

    results = asyncio.run(cron_scheduler_module.dispatch_due_crons(limit=10, scheduler_id="scheduler-1", now=due_at))
    assert len(results) == 1
    assert results[0].status == "queued"

    user_threads = asyncio.run(_list_threads_for_user("owner"))
    stateless_threads = [t for t in user_threads if t.metadata_json.get("cron_id") == cron_id]
    assert len(stateless_threads) == 1
    persisted = asyncio.run(_fetch_thread(stateless_threads[0].thread_id))
    assert persisted is not None


def test_dispatch_due_crons_never_deletes_caller_owned_thread(client: TestClient) -> None:
    from agentseek_api.services import cron_scheduler as cron_scheduler_module

    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client, user_id="owner")
    created = client.post(
        f"/threads/{thread_id}/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=MINUTELY;INTERVAL=1", "input": {"kind": "thread-bound"}},
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    due_at = datetime.now(UTC) - timedelta(minutes=1)
    asyncio.run(_mark_cron_due(cron_id, when=due_at))

    results = asyncio.run(cron_scheduler_module.dispatch_due_crons(limit=10, scheduler_id="scheduler-1", now=due_at))
    assert len(results) == 1
    assert results[0].status == "queued"

    persisted = asyncio.run(_fetch_thread(thread_id))
    assert persisted is not None
```
- [ ] **Step 2: Run them; expect failure**
Run: `uv run pytest tests/integration/test_scheduler_runtime.py::test_dispatch_due_crons_deletes_stateless_thread_on_run_completed_delete tests/integration/test_scheduler_runtime.py::test_dispatch_due_crons_keeps_stateless_thread_on_run_completed_keep tests/integration/test_scheduler_runtime.py::test_dispatch_due_crons_never_deletes_caller_owned_thread -v`
Expected: FAIL — the `delete` test fails (stateless thread still present); `keep` and caller-owned pass already (regression guards).

- [ ] **Step 3: Add imports + a module logger to `cron_scheduler.py`** — the file currently has `from sqlalchemy import select`, `from agentseek_api.core.orm import CronJob, CronTick, Run, Thread`, NO `logging`, NO logger. Replace the import block with:
```python
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import CronJob, CronTick, Run, Thread
from agentseek_api.models.api import ThreadCreate
from agentseek_api.models.auth import User
from agentseek_api.services.cron_models import ClaimedCron, CronDispatchResult
from agentseek_api.services.cron_rrule import compute_next_run_at
from agentseek_api.services.cron_webhooks import (
    build_webhook_payload,
    deliver_webhook_with_retries,
    get_webhook_http_client,
)
from agentseek_api.services.redis_queue import RedisRunQueue
from agentseek_api.services.run_preparation import (
    ActiveThreadRunConflictError,
    prepare_run,
    submit_existing_run,
)
from agentseek_api.services.stream_persistence import (
    delete_run_stream_events,
    delete_thread_stream_events,
)
from agentseek_api.services.thread_protocol import thread_protocol_broker
from agentseek_api.services.thread_service import create_thread_for_user
from agentseek_api.settings import settings

logger = logging.getLogger(__name__)

TERMINAL_RUN_STATUSES = {"success", "error", "interrupted"}
TERMINAL_TICK_STATUSES = {"success", "error", "interrupted", "skipped"}
```
- [ ] **Step 4: Add a local best-effort checkpointer helper and stateless-thread deletion routine** — add these module-level functions just after `_as_utc` in `src/agentseek_api/services/cron_scheduler.py`:
```python
async def _best_effort_checkpointer_call(method_name: str, *args: object, **kwargs: object) -> None:
    method = getattr(db_manager.get_langgraph_checkpointer(), method_name, None)
    if method is None:
        return
    try:
        result = method(*args, **kwargs)
        if hasattr(result, "__await__"):
            await result
    except NotImplementedError:
        return


async def _delete_stateless_thread(thread_id: str) -> None:
    try:
        session_factory = db_manager.get_session_factory()
        async with session_factory() as session:
            row = await session.scalar(select(Thread).where(Thread.thread_id == thread_id))
            if row is None:
                return
            run_ids = (
                await session.scalars(select(Run.run_id).where(Run.thread_id == thread_id))
            ).all()
            await session.execute(delete(Run).where(Run.thread_id == thread_id))
            await session.delete(row)
            await session.commit()
        await _best_effort_checkpointer_call("adelete_thread", thread_id)
        if run_ids:
            await _best_effort_checkpointer_call("adelete_for_runs", list(run_ids))
            await delete_run_stream_events(list(run_ids))
        thread_protocol_broker.delete_thread(thread_id)
        await delete_thread_stream_events(thread_id)
    except Exception:
        logger.exception("Failed to delete stateless cron thread %s", thread_id)
```
- [ ] **Step 5: Collect stateless threads during the reconcile loop and delete after webhook delivery** — edit `_reconcile_terminal_ticks`. Add a `threads_to_delete` list before the loop, populate it inside the `if tick.status in TERMINAL_TICK_STATUSES:` block (stateless crons only, `on_run_completed == "delete"`, real `tick.thread_id`, non-skipped status), and run deletions after webhook deliveries:
```python
        deliveries: list[tuple[int, str, int, dict[str, object]]] = []
        threads_to_delete: list[str] = []
        for tick in ticks:
            cron = await session.scalar(select(CronJob).where(CronJob.cron_id == tick.cron_id))
            if cron is None:
                continue
            if tick.status == "queued" and tick.run_id is not None:
                run = await session.scalar(select(Run).where(Run.run_id == tick.run_id))
                if run is not None and run.status in TERMINAL_RUN_STATUSES:
                    tick.status = run.status
                    tick.skip_reason = run.last_error if run.status == "error" else None
                    tick.updated_at = current_time
            if tick.status in TERMINAL_TICK_STATUSES:
                cron.last_tick_status = tick.status
                cron.last_error = tick.skip_reason if tick.status == "error" else None
                if tick.status != "skipped":
                    cron.last_run_at = current_time
                if (
                    cron.thread_id is None
                    and cron.on_run_completed == "delete"
                    and tick.thread_id is not None
                    and tick.status != "skipped"
                ):
                    threads_to_delete.append(tick.thread_id)
            delivery_is_available = tick.webhook_delivery_status is None or (
                tick.webhook_delivery_status == "delivering" and _as_utc(tick.updated_at) <= stale_before
            )
            if cron.webhook and tick.status in TERMINAL_TICK_STATUSES and delivery_is_available:
                tick.webhook_delivery_status = "delivering"
                deliveries.append(
                    (
                        tick.id,
                        cron.webhook,
                        cron.max_webhook_attempts,
                        build_webhook_payload(cron=cron, tick=tick),
                    )
                )
        await session.commit()

    webhook_client = http_client or get_webhook_http_client()
    for tick_id, webhook_url, max_attempts, payload in deliveries:
        await deliver_webhook_with_retries(
            webhook_url=webhook_url,
            payload=payload,
            tick_id=tick_id,
            max_attempts=max_attempts,
            http_client=webhook_client,
            sleep=sleep,
        )

    for thread_id in threads_to_delete:
        await _delete_stateless_thread(thread_id)
```
- [ ] **Step 6: Fix the pre-existing stateless test broken by the new `delete` default**
`test_dispatch_due_crons_creates_stateless_run_and_skips_busy_thread` creates a stateless cron WITHOUT `on_run_completed`, which now defaults to `"delete"` — so its thread/run get deleted and the lines ~167-180 assertions (`len(stateless_threads) == 1`, run inspection) would fail. Pin that test to the old behavior by adding `"on_run_completed": "keep"` to its stateless cron POST body (the `/runs/crons` call around line 127-138):
```python
    stateless = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=1",
            "input": {"kind": "stateless"},
            "metadata": {"source": "scheduler-runtime"},
            "config": {"model": "gpt-test"},
            "context": {"tenant": "acme"},
            "on_run_completed": "keep",
        },
        headers={"x-user-id": "owner"},
    )
```
- [ ] **Step 7: Run the new + fixed tests; expect pass**
Run: `uv run pytest tests/integration/test_scheduler_runtime.py::test_dispatch_due_crons_deletes_stateless_thread_on_run_completed_delete tests/integration/test_scheduler_runtime.py::test_dispatch_due_crons_keeps_stateless_thread_on_run_completed_keep tests/integration/test_scheduler_runtime.py::test_dispatch_due_crons_never_deletes_caller_owned_thread tests/integration/test_scheduler_runtime.py::test_dispatch_due_crons_creates_stateless_run_and_skips_busy_thread -v`
Expected: PASS.

- [ ] **Step 8: Run the full integration scheduler runtime suite**
Run: `uv run pytest tests/integration/test_scheduler_runtime.py -v`
Expected: PASS — all green.

- [ ] **Step 9: Commit**
```bash
git add src/agentseek_api/services/cron_scheduler.py tests/integration/test_scheduler_runtime.py
git commit -m "feat(crons): delete ephemeral stateless thread when on_run_completed is delete"
```

---

### Task 11 — Full suite green + self-review
**Files:** none (verification + docs sanity only)

- [ ] **Step 1: Run the entire cron + scheduler test surface**
Run: `uv run pytest tests/unit/test_cron_service.py tests/unit/test_scheduler.py tests/unit/test_cron_rrule.py tests/integration/test_cron_api.py tests/integration/test_scheduler_runtime.py -v`
Expected: PASS — all green.

- [ ] **Step 2: Run the full project test suite to catch cross-module regressions**
Run: `uv run pytest -q`
Expected: PASS (or only pre-existing unrelated failures — compare against a clean `develop` baseline if anything fails).

- [ ] **Step 3: Lint/type sanity (if configured)**
Run: `uv run ruff check src/agentseek_api/models/api.py src/agentseek_api/services/cron_service.py src/agentseek_api/services/cron_scheduler.py src/agentseek_api/services/cron_models.py src/agentseek_api/api/crons.py src/agentseek_api/core/orm.py`
Expected: no errors (skip if ruff is not part of the project).

- [ ] **Step 4: Spec-coverage self-check**
Confirm each issue-#44 gap is addressed: CronCreate missing fields (Task 3); ThreadCronCreate separation + multitask_strategy + no on_run_completed (Task 4); CronPatch fields (Task 5); CronRead user_id/payload/end_time/metadata + next_run_date rename (Task 2); CronSearch metadata/sort_by/sort_order/select + limit bounds (Task 6); CronCount metadata (Task 6); end_time enforcement (Task 8); on_run_completed behavior (Task 10). The extra GET endpoint and extension fields are intentionally kept per the approved spec.

- [ ] **Step 5: Final verification statement**
Report the exact `uv run pytest -q` summary line (passed/failed counts) as evidence. Do not claim completion without it.
