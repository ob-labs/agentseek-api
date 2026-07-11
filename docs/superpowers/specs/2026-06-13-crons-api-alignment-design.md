# Crons API Alignment with LangGraph Platform Spec (#44)

**Date:** 2026-06-13
**Issue:** #44 — "[Bug]: The crons API is not fully aligned"
**Branch base:** `develop`

## Goal

Bring the Crons API into alignment with the LangGraph Platform OpenAPI spec by
adding all missing request/response fields and wiring their behavior, while
**keeping** the project's existing working extensions (webhook delivery,
timezone handling, `last_run_at`/`last_tick_status`/`last_error` tracking, and
the `GET /runs/crons/{cron_id}` endpoint).

## Decisions (confirmed with user)

1. **Posture: additive superset.** Add every missing spec field AND keep the
   non-standard extensions. Non-breaking.
2. **Behavior scope: full.** Wire run-control passthrough fields (group A),
   `end_time` enforcement, AND `on_run_completed` thread deletion (group B).
3. **`next_run_at` → `next_run_date`: expose both.** Response carries the spec
   name `next_run_date` and keeps `next_run_at` populated as a deprecated
   extension. Fully non-breaking.
4. **Keep `GET /runs/crons/{cron_id}`** as a non-standard extension.

## Non-Goals

- No Alembic migrations: schema is applied via `Base.metadata.create_all`
  (see `core/database.py`). New columns are added directly to the ORM model.
- No change to the webhook delivery subsystem behavior.
- No change to the RRULE parser (`cron_rrule.py`); `end_time` is enforced by the
  scheduler, independent of the RRULE `UNTIL` clause.

## Spec field inventory

| Field | CronCreate | ThreadCronCreate | CronPatch | CronRead | Storage |
|-------|:---:|:---:|:---:|:---:|---|
| `end_time` | ✅ | ✅ | ✅ | ✅ | new ORM column |
| `interrupt_before` | ✅ | ✅ | ✅ | — | `kwargs_json` |
| `interrupt_after` | ✅ | ✅ | ✅ | — | `kwargs_json` |
| `on_run_completed` | ✅ (`delete`) | ❌ (spec omits) | ✅ | — | new ORM column |
| `stream_mode` | ✅ (`["values"]`) | ✅ | ✅ | — | `kwargs_json` |
| `stream_subgraphs` | ✅ (`false`) | ✅ | ✅ | — | `kwargs_json` |
| `stream_resumable` | ✅ (`false`) | ✅ | ✅ | — | `kwargs_json` |
| `durability` | ✅ (`async`) | ✅ | ✅ | — | `kwargs_json` |
| `multitask_strategy` | ❌ (spec omits) | ✅ (`enqueue`) | — | — | `kwargs_json` |
| `user_id` | — | — | — | ✅ | existing column |
| `payload` | — | — | — | ✅ | derived |
| `metadata` | ✅ | ✅ | ✅ | ✅ | existing column |

Extension fields kept on `CronRead`: `timezone`, `webhook`, `last_run_at`,
`last_tick_status`, `last_error`, `enabled`, `next_run_at` (alias of
`next_run_date`).

## Section 1 — Schema changes (`models/api.py`)

### 1a. New type aliases
Reuse existing run-control aliases: `RunInterrupt`, `RunDurability`,
`RunStreamMode`, `RunMultitaskStrategy`. Add:

```python
CronOnRunCompleted = Literal["delete", "keep"]
CronSortBy = Literal[
    "cron_id", "assistant_id", "thread_id", "next_run_date",
    "end_time", "created_at", "updated_at",
]
CronSortOrder = Literal["asc", "desc"]
CronSelectField = Literal[ ...CronRead field names... ]
```

### 1b. `CronCreate` (stateless)
- `model_config`: `extra="forbid"` → `extra="allow"` (spec uses
  `additionalProperties: true`).
- `assistant_id: str` unchanged — `resolve_assistant_id` already accepts both a
  UUID and a graph name at runtime, which covers the spec's `anyOf(UUID,string)`.
- Add: `end_time: datetime | None = None`, `interrupt_before: RunInterrupt | None = None`,
  `interrupt_after: RunInterrupt | None = None`,
  `on_run_completed: CronOnRunCompleted = "delete"`,
  `stream_mode: RunStreamMode | list[RunStreamMode] | None = Field(default_factory=lambda: ["values"])`,
  `stream_subgraphs: bool = False`, `stream_resumable: bool = False`,
  `durability: RunDurability = "async"`.
- Keep extensions: `timezone`, `webhook`, `enabled`.

### 1c. New `ThreadCronCreate`
Same as `CronCreate` **minus** `on_run_completed`, **plus**
`multitask_strategy: RunMultitaskStrategy = "enqueue"`. The two thread-cron
endpoints (`POST /threads/{thread_id}/runs/crons`) switch to this type.

### 1d. `CronPatch`
Add optional, `model_fields_set`-gated: `end_time`, `interrupt_before`,
`interrupt_after`, `on_run_completed`, `stream_mode`, `stream_subgraphs`,
`stream_resumable`, `durability`.

### 1e. `CronRead`
- Add `next_run_date: datetime` (spec). Keep `next_run_at: datetime` populated
  with the same value (deprecated extension) — both serialized.
- Add spec fields: `user_id: str | None`, `payload: dict[str, Any]`,
  `end_time: datetime | None`, `metadata: dict[str, Any]`.
- `payload` is reconstructed: `{"input": input_json, "config": <from kwargs>,
  "context": <from kwargs>}`.
- Keep extension fields: `timezone`, `webhook`, `last_run_at`,
  `last_tick_status`, `last_error`.

### 1f. `CronSearchRequest`
Add `metadata: dict | None`, `sort_by: CronSortBy | None`,
`sort_order: CronSortOrder | None`, `select: list[CronSelectField] | None`.
Change `limit` from `Field(default=10, ge=0)` to
`Field(default=10, ge=1, le=1000)` (matches Assistant/Thread search). Keep
`enabled` extension. Verified: no cron test sends `limit=0`, so this is safe.

### 1g. `CronCountRequest`
Add `metadata: dict | None`. Keep `enabled` extension.

## Section 2 — Data model (`core/orm.py`)

Add to `CronJob` (nullable / defaulted, created via `create_all`):
- `end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)`
- `on_run_completed: Mapped[str] = mapped_column(String(16), nullable=False, default="delete")`

Run-control fields (`stream_mode`, `interrupt_before/after`, `durability`,
`stream_subgraphs`, `stream_resumable`, `multitask_strategy`) are stored in the
existing `kwargs_json` blob — **no new columns** — mirroring how `runs.py`
builds `run_kwargs`. This keeps the dispatch path unchanged.

## Section 3 — Service layer (`cron_service.py`)

- Extend `_cron_kwargs` to fold in run-control fields, storing only non-default
  values (mirroring `runs.py`), under `kwargs_json`. Proposed shape:
  `{"config": ..., "context": ..., "stream_modes": [...], "interrupt_before": ...,
  "interrupt_after": ..., "durability": ..., "stream_subgraphs": ...,
  "stream_resumable": ..., "multitask_strategy": ...}`.
- `create_cron` gains the new params; sets `end_time` and `on_run_completed`
  columns; stores run-control in kwargs.
- `_to_read_model` populates `next_run_date` (+ `next_run_at` alias), `payload`,
  `user_id`, `metadata`, `end_time`, and existing extension fields.
- `patch_cron` handles new `model_fields_set` keys (run-control merged into
  existing kwargs, columns for `end_time`/`on_run_completed`).
- `search_crons`: apply `metadata` exact-match filter, `sort_by`/`sort_order`
  (default `created_at`/`desc`; `next_run_date` maps to `next_run_at` column),
  and `select` projection.
- `count_crons`: apply `metadata` filter.
- `select` projection follows the assistants/threads pattern:
  `response_model_exclude_none=True` on the route + returning only requested
  fields.

## Section 4 — Scheduler behavior (`cron_scheduler.py`, `cron_models.py`)

### `ClaimedCron` (`cron_models.py`)
Add `on_run_completed: str`, `end_time: datetime | None`, and
`multitask_strategy: str` (read from `kwargs_json`). Populate in `from_row`.

### `end_time` enforcement (`claim_due_crons`)
- Due-query gains: `(CronJob.end_time.is_(None)) | (CronJob.end_time > current_time)`.
- After `compute_next_run_at`, if the new `next_run_at` exceeds `end_time`, set
  `row.enabled = False` (cron exhausted) — same branch as the `ValueError` path.

### `on_run_completed` deletion (`_reconcile_terminal_ticks`)
- **Stateless vs thread cron** is determined by the authoritative signal
  `CronJob.thread_id IS NULL` (stateless crons store no thread_id; the ephemeral
  thread is created per-fire in `dispatch_claimed_cron`). The per-fire thread to
  delete is `tick.thread_id`.
- When a tick first transitions to a terminal run status, the cron is stateless
  (`CronJob.thread_id IS NULL`), `on_run_completed == "delete"`, and
  `tick.thread_id` is set: delete that ephemeral thread and its runs, reusing the
  same cleanup steps as `delete_thread` (delete `Run` rows, delete `Thread`,
  best-effort checkpointer cleanup, stream-event cleanup). Deletion happens
  **after** webhook payload is built so the webhook still reports `thread_id`.
- **Thread crons** (`CronJob.thread_id` set) reuse a caller-owned thread; the
  spec omits `on_run_completed` for them, so deletion is always skipped.
- `keep` retains the thread (current behavior).
- Deletion is best-effort: failures are logged, not raised, so tick
  reconciliation and webhook delivery are never blocked by cleanup errors.

### Dispatch (`dispatch_claimed_cron`)
- Pass `multitask_strategy=claim.multitask_strategy` into `prepare_run` for
  thread crons.

## Section 5 — Tests

**`tests/integration/test_cron_api.py`:**
- Round-trip all new `CronCreate` fields; assert `next_run_date` present and
  `next_run_at` alias equal.
- `ThreadCronCreate` accepts `multitask_strategy`; rejects `on_run_completed`
  (422).
- Search with `sort_by`, `sort_order`, `metadata`, `select`.
- Count with `metadata`.

**`tests/unit/test_cron_service.py`:**
- Update for new `_to_read_model` shape (`payload`, `next_run_date`, etc.).
- Run-control fields land in `kwargs_json`.

**`tests/unit/test_scheduler.py` / `tests/integration/test_scheduler_runtime.py`:**
- `end_time` in the past → not claimed; crossing `end_time` → disabled.
- Stateless `on_run_completed="delete"` → ephemeral thread deleted after
  terminal run; `"keep"` → retained.
- Thread cron → thread never deleted regardless of setting.

## Risk / Compatibility

- **Non-breaking:** all new request fields are optional with spec defaults; all
  new response fields are additive; `next_run_at` retained.
- **`limit` tightening** (`ge=0` → `ge=1`): verified no cron caller uses
  `limit=0`.
- **`extra="allow"`** loosens stateless `CronCreate` validation — intentional,
  matches spec `additionalProperties: true`. `ThreadCronCreate` mirrors this.
- **`on_run_completed` default is `delete`** (per spec) — stateless crons will
  now clean up their ephemeral threads by default. This changes current
  behavior (threads previously accumulated), but is the spec-correct default and
  reduces thread litter. Called out explicitly for review.
