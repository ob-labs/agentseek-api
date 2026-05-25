# Cron Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add workable scheduler-backed cron support to AgentSeek API for both stateless and thread-bound cron modes, including CRUD/search/count endpoints, dedicated scheduler execution, and webhook delivery with bounded retries.

**Architecture:** Persist cron definitions, fired ticks, and webhook attempts in the existing metadata database. Add a dedicated scheduler process that uses a Redis leader lease to claim due crons and dispatches them into the existing run execution path while preserving the one-active-run-per-thread invariant by skipping busy thread-bound ticks.

**Tech Stack:** FastAPI, Pydantic, SQLAlchemy async ORM, Redis async client, existing AgentSeek executor/worker infrastructure, pytest, RRULE parsing via `dateutil.rrule`

---

## File Structure

- Create: `src/agentseek_api/api/crons.py`
  - FastAPI router for cron CRUD/search/count endpoints.
- Create: `src/agentseek_api/services/cron_models.py`
  - Internal dataclasses and enums for scheduler/tick/webhook flows.
- Create: `src/agentseek_api/services/cron_rrule.py`
  - RRULE parsing, validation, timezone normalization, and next-occurrence helpers.
- Create: `src/agentseek_api/services/cron_service.py`
  - Cron CRUD/search/count persistence and conversion to API models.
- Create: `src/agentseek_api/services/cron_scheduler.py`
  - Due-cron claiming, tick creation, stateless/thread-bound dispatch, skip-on-busy handling.
- Create: `src/agentseek_api/services/cron_webhooks.py`
  - Webhook payload building, delivery, retry loop, and persisted attempt records.
- Create: `src/agentseek_api/scheduler.py`
  - Dedicated scheduler process entrypoint with Redis leader lease.
- Modify: `src/agentseek_api/core/orm.py`
  - Add `CronJob`, `CronTick`, and `CronWebhookAttempt` tables.
- Modify: `src/agentseek_api/models/api.py`
  - Add cron request/response/search/count models.
- Modify: `src/agentseek_api/main.py`
  - Register cron router, set `crons: true`, remove cron from unsupported features.
- Modify: `src/agentseek_api/settings.py`
  - Add scheduler and webhook retry settings / Redis lease keys.
- Modify: `src/agentseek_api/cli.py`
  - Add `scheduler` command and runtime env wiring.
- Modify: `src/agentseek_api/services/redis_queue.py`
  - Extract or generalize Redis lease helpers so worker and scheduler can both use them safely.
- Modify: `tests/unit/test_cli.py`
  - Cover new `scheduler` command behavior.
- Modify: `tests/integration/test_system_endpoints.py`
  - Flip `/info` assertions for cron support.
- Create: `tests/unit/test_cron_rrule.py`
  - Validate RRULE acceptance and rejection behavior.
- Create: `tests/unit/test_cron_service.py`
  - Cover CRUD/search/count and API model conversion.
- Create: `tests/unit/test_cron_webhooks.py`
  - Cover payload shape, retry policy, and persistence.
- Create: `tests/unit/test_scheduler.py`
  - Cover due-cron claiming, stateless dispatch, thread-bound skip behavior, and leader lease handling.
- Create: `tests/integration/test_cron_api.py`
  - End-to-end cron route coverage.
- Create: `tests/integration/test_scheduler_runtime.py`
  - End-to-end due cron dispatch and webhook attempt persistence.
- Modify: `README.md`
  - Document cron support, deployment topology, and remaining limits.

### Task 1: Persist Cron Resources and Expose API Models

**Files:**
- Create: `src/agentseek_api/api/crons.py`
- Modify: `src/agentseek_api/core/orm.py`
- Modify: `src/agentseek_api/models/api.py`
- Test: `tests/integration/test_cron_api.py`

- [ ] **Step 1: Write the failing API shape tests**

```python
from fastapi.testclient import TestClient


def test_create_stateless_cron_persists_next_run_and_returns_resource(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "cron", "graph_id": "default"}).json()

    response = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant["assistant_id"],
            "input": {"message": "hello"},
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "timezone": "UTC",
            "enabled": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_id"] == assistant["assistant_id"]
    assert body["thread_id"] is None
    assert body["enabled"] is True
    assert body["schedule"] == "FREQ=MINUTELY;INTERVAL=5"
    assert body["next_run_at"]


def test_create_thread_bound_cron_rejects_missing_thread(client: TestClient) -> None:
    assistant = client.post("/assistants", json={"name": "cron", "graph_id": "default"}).json()

    response = client.post(
        "/threads/missing-thread/runs/crons",
        json={
            "assistant_id": assistant["assistant_id"],
            "input": {"message": "hello"},
            "schedule": "FREQ=DAILY;INTERVAL=1",
            "enabled": True,
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Thread not found"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_cron_api.py -k "create_stateless_cron or create_thread_bound_cron" -q`

Expected: FAIL with `404` on missing routes or import errors for missing cron models/router.

- [ ] **Step 3: Add ORM and API models**

```python
class CronJob(Base):
    __tablename__ = "cron_jobs"

    cron_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    thread_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    assistant_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    input_json: Mapped[dict] = mapped_column("input", JSON, default=dict, nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    kwargs_json: Mapped[dict] = mapped_column("kwargs", JSON, default=dict, nullable=False)
    schedule_rrule: Mapped[str] = mapped_column(String(1024), nullable=False)
    timezone: Mapped[str] = mapped_column(String(128), nullable=False, default="UTC")
    enabled: Mapped[bool] = mapped_column(nullable=False, default=True)
    webhook: Mapped[str | None] = mapped_column(Text, nullable=True)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_tick_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    max_webhook_attempts: Mapped[int] = mapped_column(nullable=False, default=3)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False)


class CronCreate(BaseModel):
    assistant_id: str
    input: Any
    metadata: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    schedule: str
    timezone: str | None = None
    webhook: str | None = None
    enabled: bool = True


class CronRead(BaseModel):
    cron_id: str
    assistant_id: str
    thread_id: str | None = None
    schedule: str
    timezone: str
    enabled: bool
    webhook: str | None = None
    next_run_at: datetime | None = None
    last_run_at: datetime | None = None
    last_tick_status: str | None = None
    last_error: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
```

- [ ] **Step 4: Add minimal router skeleton**

```python
router = APIRouter(prefix="/runs/crons", tags=["Crons"])


@router.post("", response_model=CronRead)
async def create_cron(payload: CronCreate, user: User = Depends(get_current_user)) -> CronRead:
    return await cron_service.create_stateless_cron(payload=payload, user=user)


@thread_router.post("/threads/{thread_id}/runs/crons", response_model=CronRead)
async def create_thread_cron(thread_id: str, payload: CronCreate, user: User = Depends(get_current_user)) -> CronRead:
    return await cron_service.create_thread_cron(thread_id=thread_id, payload=payload, user=user)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_cron_api.py -k "create_stateless_cron or create_thread_bound_cron" -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/agentseek_api/core/orm.py src/agentseek_api/models/api.py src/agentseek_api/api/crons.py tests/integration/test_cron_api.py
git commit -m "feat: add cron persistence models and create routes"
```

### Task 2: Add RRULE Validation and Cron CRUD/Search/Count Service

**Files:**
- Create: `src/agentseek_api/services/cron_rrule.py`
- Create: `src/agentseek_api/services/cron_service.py`
- Modify: `src/agentseek_api/api/crons.py`
- Test: `tests/unit/test_cron_rrule.py`
- Test: `tests/unit/test_cron_service.py`
- Test: `tests/integration/test_cron_api.py`

- [ ] **Step 1: Write failing validation and CRUD tests**

```python
import pytest

from agentseek_api.services.cron_rrule import compute_next_run_at, validate_schedule


def test_validate_schedule_rejects_unsupported_clause() -> None:
    with pytest.raises(ValueError, match="Unsupported RRULE clause: BYSETPOS"):
        validate_schedule("FREQ=MONTHLY;BYSETPOS=1", timezone_name="UTC")


def test_compute_next_run_at_returns_future_utc_datetime() -> None:
    next_run = compute_next_run_at("FREQ=MINUTELY;INTERVAL=5", timezone_name="UTC")
    assert next_run is not None
    assert next_run.tzinfo is not None


def test_search_crons_filters_by_assistant_and_enabled(client: TestClient) -> None:
    response = client.post("/runs/crons/search", json={"assistant_id": "a1", "enabled": True, "limit": 10, "offset": 0})
    assert response.status_code == 200
    assert "items" in response.json()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_cron_rrule.py tests/unit/test_cron_service.py tests/integration/test_cron_api.py -k "schedule or search_crons" -q`

Expected: FAIL with missing module/function errors or incorrect route behavior.

- [ ] **Step 3: Implement RRULE parsing and service helpers**

```python
SUPPORTED_RRULE_KEYS = {"FREQ", "INTERVAL", "BYDAY", "BYHOUR", "BYMINUTE", "BYMONTHDAY", "COUNT", "UNTIL"}


def validate_schedule(schedule: str, *, timezone_name: str) -> None:
    parts = [chunk for chunk in schedule.split(";") if chunk]
    seen_keys: set[str] = set()
    for part in parts:
        key, _, value = part.partition("=")
        if not key or not value:
            raise ValueError("Malformed RRULE")
        normalized_key = key.upper()
        if normalized_key not in SUPPORTED_RRULE_KEYS:
            raise ValueError(f"Unsupported RRULE clause: {normalized_key}")
        seen_keys.add(normalized_key)
    if "FREQ" not in seen_keys:
        raise ValueError("RRULE must include FREQ")
    ZoneInfo(timezone_name)


async def create_stateless_cron(*, payload: CronCreate, user: User) -> CronRead:
    timezone_name = payload.timezone or "UTC"
    validate_schedule(payload.schedule, timezone_name=timezone_name)
    next_run_at = compute_next_run_at(payload.schedule, timezone_name=timezone_name)
    cron = CronJob(
        user_id=user.identity,
        assistant_id=payload.assistant_id,
        input_json=_coerce_input(payload.input),
        metadata_json=payload.metadata,
        kwargs_json={"config": payload.config, "context": payload.context},
        schedule_rrule=payload.schedule,
        timezone=timezone_name,
        enabled=payload.enabled,
        webhook=payload.webhook,
        next_run_at=next_run_at,
    )
```

- [ ] **Step 4: Add router methods for get, patch, delete, search, and count**

```python
@router.post("/search", response_model=CronSearchResponse)
async def search_crons(payload: CronSearchRequest, user: User = Depends(get_current_user)) -> CronSearchResponse:
    return await cron_service.search_crons(payload=payload, user=user)


@router.post("/count", response_model=CronCountResponse)
async def count_crons(payload: CronCountRequest, user: User = Depends(get_current_user)) -> CronCountResponse:
    return await cron_service.count_crons(payload=payload, user=user)


@router.patch("/{cron_id}", response_model=CronRead)
async def patch_cron(cron_id: str, payload: CronPatch, user: User = Depends(get_current_user)) -> CronRead:
    return await cron_service.patch_cron(cron_id=cron_id, payload=payload, user=user)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_cron_rrule.py tests/unit/test_cron_service.py tests/integration/test_cron_api.py -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/agentseek_api/services/cron_rrule.py src/agentseek_api/services/cron_service.py src/agentseek_api/api/crons.py tests/unit/test_cron_rrule.py tests/unit/test_cron_service.py tests/integration/test_cron_api.py
git commit -m "feat: add cron validation and CRUD service"
```

### Task 3: Add Scheduler Process and Due-Cron Dispatch

**Files:**
- Create: `src/agentseek_api/services/cron_models.py`
- Create: `src/agentseek_api/services/cron_scheduler.py`
- Create: `src/agentseek_api/scheduler.py`
- Modify: `src/agentseek_api/settings.py`
- Modify: `src/agentseek_api/cli.py`
- Modify: `src/agentseek_api/services/redis_queue.py`
- Test: `tests/unit/test_scheduler.py`
- Test: `tests/unit/test_cli.py`
- Test: `tests/integration/test_scheduler_runtime.py`

- [ ] **Step 1: Write failing scheduler and CLI tests**

```python
import asyncio

import pytest

from agentseek_api.services.cron_scheduler import claim_due_crons


@pytest.mark.asyncio
async def test_claim_due_crons_returns_each_due_cron_once(db_seeded_due_crons) -> None:
    first = await claim_due_crons(limit=10, scheduler_id="s1")
    second = await claim_due_crons(limit=10, scheduler_id="s2")
    assert len(first) == 1
    assert second == []


def test_scheduler_command_uses_runtime_env_and_scheduler_module(tmp_path: Path) -> None:
    from agentseek_api.cli import main

    config_path = _write_basic_langgraph_config(tmp_path)
    capture = _RunCapture()
    exit_code = main(["scheduler", "--config", str(config_path)], runner=capture, cwd=tmp_path)
    assert exit_code == 0
    assert capture.command[1:] == ["-m", "agentseek_api.scheduler"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_scheduler.py tests/unit/test_cli.py -k "scheduler" -q`

Expected: FAIL with missing scheduler module/command or missing claim helpers.

- [ ] **Step 3: Implement scheduler lease and due-cron claim logic**

```python
class RedisLeaseClient:
    async def acquire(self, key: str, owner: str, *, ttl_seconds: int) -> bool: ...
    async def renew(self, key: str, owner: str, *, ttl_seconds: int) -> bool: ...
    async def release(self, key: str, owner: str) -> None: ...


async def claim_due_crons(*, limit: int, scheduler_id: str, now: datetime | None = None) -> list[CronJob]:
    current_time = now or datetime.now(UTC)
    async with db_manager.get_session_factory()() as session:
        rows = (
            await session.scalars(
                select(CronJob)
                .where(CronJob.enabled.is_(True), CronJob.next_run_at.is_not(None), CronJob.next_run_at <= current_time)
                .order_by(CronJob.next_run_at.asc())
                .limit(limit)
                .with_for_update()
            )
        ).all()
        for cron in rows:
            tick = CronTick(cron_id=cron.cron_id, scheduled_for=cron.next_run_at, status="started", started_at=current_time)
            session.add(tick)
            cron.next_run_at = compute_next_run_at(cron.schedule_rrule, timezone_name=cron.timezone, after=current_time)
        await session.commit()
        return rows
```

- [ ] **Step 4: Implement stateless dispatch and thread-bound skip-on-busy**

```python
async def dispatch_cron(cron: CronJob, tick: CronTick) -> None:
    if cron.thread_id is None:
        thread = await create_thread_for_cron(user_id=cron.user_id, metadata={"cron_id": cron.cron_id})
        run = await prepare_and_submit_run(
            thread_id=thread.thread_id,
            assistant_id=cron.assistant_id,
            payload=cron.input_json,
            user=User(identity=cron.user_id, is_authenticated=True),
            metadata=cron.metadata_json,
            kwargs=cron.kwargs_json,
        )
        tick.run_id = run.run_id
        tick.status = "queued"
        return

    if await thread_has_active_run(thread_id=cron.thread_id, user_id=cron.user_id):
        tick.status = "skipped"
        tick.skip_reason = "thread_busy"
        return

    run = await prepare_and_submit_run(
        thread_id=cron.thread_id,
        assistant_id=cron.assistant_id,
        payload=cron.input_json,
        user=User(identity=cron.user_id, is_authenticated=True),
        metadata=cron.metadata_json,
        kwargs=cron.kwargs_json,
    )
    tick.run_id = run.run_id
    tick.status = "queued"
```

- [ ] **Step 5: Wire CLI and process entrypoint**

```python
def build_scheduler_command() -> list[str]:
    return [sys.executable, "-m", "agentseek_api.scheduler"]


if command == "scheduler":
    return _execute_scheduler_command(args, runner=run, cwd=workdir)
```

Run: `uv run pytest tests/unit/test_scheduler.py tests/unit/test_cli.py -k "scheduler" -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/agentseek_api/services/cron_models.py src/agentseek_api/services/cron_scheduler.py src/agentseek_api/scheduler.py src/agentseek_api/settings.py src/agentseek_api/cli.py src/agentseek_api/services/redis_queue.py tests/unit/test_scheduler.py tests/unit/test_cli.py tests/integration/test_scheduler_runtime.py
git commit -m "feat: add cron scheduler process"
```

### Task 4: Add Webhook Delivery Attempts and Retry Persistence

**Files:**
- Create: `src/agentseek_api/services/cron_webhooks.py`
- Modify: `src/agentseek_api/services/cron_scheduler.py`
- Test: `tests/unit/test_cron_webhooks.py`
- Test: `tests/integration/test_scheduler_runtime.py`

- [ ] **Step 1: Write failing webhook tests**

```python
import pytest

from agentseek_api.services.cron_webhooks import deliver_webhook_with_retries


@pytest.mark.asyncio
async def test_deliver_webhook_with_retries_records_each_attempt(fake_http_client, persisted_tick) -> None:
    fake_http_client.failures_before_success = 2
    result = await deliver_webhook_with_retries(
        webhook_url="https://example.com/hook",
        payload={"cron_id": "c1", "status": "success"},
        tick_id=persisted_tick.tick_id,
        max_attempts=3,
        http_client=fake_http_client,
    )
    assert result.delivered is True
    assert result.attempt_count == 3
    assert result.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_cron_webhooks.py tests/integration/test_scheduler_runtime.py -k "webhook" -q`

Expected: FAIL with missing webhook module or missing attempt persistence.

- [ ] **Step 3: Implement webhook delivery and persistence**

```python
async def deliver_webhook_with_retries(
    *,
    webhook_url: str,
    payload: dict[str, object],
    tick_id: str,
    max_attempts: int,
    http_client: AsyncWebhookClient,
) -> WebhookDeliveryResult:
    for attempt_number in range(1, max_attempts + 1):
        try:
            response = await http_client.post(webhook_url, json=payload)
            await persist_webhook_attempt(tick_id=tick_id, attempt_number=attempt_number, status_code=response.status_code, error=None)
            if 200 <= response.status_code < 300:
                return WebhookDeliveryResult(delivered=True, attempt_count=attempt_number, status_code=response.status_code)
        except Exception as exc:
            await persist_webhook_attempt(tick_id=tick_id, attempt_number=attempt_number, status_code=None, error=str(exc))
        await asyncio.sleep(min(2 ** (attempt_number - 1), 8))
    return WebhookDeliveryResult(delivered=False, attempt_count=max_attempts, status_code=None)
```

- [ ] **Step 4: Update scheduler reconciliation to trigger webhook delivery on terminal tick/run status**

```python
if cron.webhook and tick.status in {"success", "error", "skipped"}:
    await deliver_webhook_with_retries(
        webhook_url=cron.webhook,
        payload=build_webhook_payload(cron=cron, tick=tick),
        tick_id=tick.tick_id,
        max_attempts=cron.max_webhook_attempts,
        http_client=http_client,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_cron_webhooks.py tests/integration/test_scheduler_runtime.py -k "webhook" -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/agentseek_api/services/cron_webhooks.py src/agentseek_api/services/cron_scheduler.py tests/unit/test_cron_webhooks.py tests/integration/test_scheduler_runtime.py
git commit -m "feat: add cron webhook retries"
```

### Task 5: Flip Public Capability Surface and Document Deployment

**Files:**
- Modify: `src/agentseek_api/main.py`
- Modify: `tests/integration/test_system_endpoints.py`
- Modify: `README.md`
- Test: `tests/integration/test_system_endpoints.py`
- Test: `tests/integration/test_cron_api.py`
- Test: `tests/integration/test_scheduler_runtime.py`

- [ ] **Step 1: Write the failing system endpoint assertions**

```python
def test_info_endpoint_reports_crons_supported(client: TestClient) -> None:
    response = client.get("/info")
    body = response.json()
    assert body["flags"]["crons"] is True
    assert "crons" not in body["metadata"]["unsupported_features"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_system_endpoints.py -k "crons_supported" -q`

Expected: FAIL because `/info` still reports `crons: false`.

- [ ] **Step 3: Flip runtime flags and update docs**

```python
def _feature_flags(*, a2a_enabled: bool, mcp_enabled: bool) -> dict[str, bool]:
    return {
        "agents": True,
        "assistants": True,
        "threads": True,
        "runs": True,
        "crons": True,
        "store": True,
        "a2a": a2a_enabled,
        "mcp": mcp_enabled,
        "protocol_v2": True,
    }
```

```markdown
- Implemented: assistants, threads, runs, crons, streaming, Store API, MCP, and A2A
- Deployment roles for cron support: API server, worker, scheduler
- Remaining intentional gaps: distributed runtime parity, assistant subgraph inspection, assistant version promotion
```

- [ ] **Step 4: Run focused verification**

Run: `uv run pytest tests/integration/test_system_endpoints.py tests/integration/test_cron_api.py tests/integration/test_scheduler_runtime.py -q`

Expected: PASS

- [ ] **Step 5: Run final broader verification**

Run: `uv run pytest tests/unit/test_cron_rrule.py tests/unit/test_cron_service.py tests/unit/test_scheduler.py tests/unit/test_cron_webhooks.py tests/unit/test_cli.py tests/integration/test_cron_api.py tests/integration/test_scheduler_runtime.py tests/integration/test_system_endpoints.py -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/agentseek_api/main.py tests/integration/test_system_endpoints.py README.md
git commit -m "docs: publish cron capability surface"
```

## Self-Review

### Spec coverage

- Cron API routes: covered by Task 1 and Task 2.
- RRULE validation and explicit unsupported-clause rejection: covered by Task 2.
- Dedicated scheduler with Redis leader lease: covered by Task 3.
- Stateless dispatch and thread-bound skip-on-busy semantics: covered by Task 3.
- Webhook bounded retries with persisted attempts: covered by Task 4.
- `/info` and README capability updates: covered by Task 5.

No spec gaps remain.

### Placeholder scan

- No `TODO`, `TBD`, or deferred “implement later” language remains in the task steps.
- Every code-changing step includes concrete code snippets and exact files.
- Every verification step includes exact commands and expected outcomes.

### Type consistency

- Cron persistence model names stay consistent as `CronJob`, `CronTick`, and `CronWebhookAttempt`.
- API models stay consistent as `CronCreate`, `CronPatch`, `CronRead`, `CronSearchRequest`, `CronSearchResponse`, `CronCountRequest`, and `CronCountResponse`.
- Scheduler and webhook helpers consistently refer to `schedule`, `timezone`, `next_run_at`, `tick_id`, and `max_webhook_attempts`.
