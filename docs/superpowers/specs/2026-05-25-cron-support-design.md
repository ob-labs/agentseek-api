# AgentSeek API Cron Support Design

## Goal

Add workable OSS cron parity for AgentSeek API that is useful to the community without claiming full LangSmith deployment parity.

This work is explicitly scoped to:

- both LangSmith-style cron creation modes
  - stateless crons via `POST /runs/crons`
  - thread-bound crons via `POST /threads/{thread_id}/runs/crons`
- persisted cron records with search, count, read, update, and delete endpoints
- a real scheduler process that dispatches due cron jobs
- webhook delivery on cron completion with bounded retries
- RRULE-shaped schedule input with explicit validation and rejection of unsupported clauses

This work is explicitly out of scope:

- full distributed runtime orchestration parity
- high-availability multi-scheduler sharding beyond single-leader election
- transactional or exactly-once webhook delivery guarantees
- replaying missed cron fires across long outages beyond the persisted next-fire contract

## Release Standard

The release should be safe to present as:

- workable cron support for the core Agent Server-compatible surface
- compatible with both thread-bound and stateless cron creation flows
- backed by a real scheduler, real run dispatch, and real webhook delivery attempts

The release should not imply:

- full LangSmith deployment semantics
- active-active scheduler execution
- full RRULE feature coverage if certain clauses remain unsupported
- guaranteed webhook delivery or cryptographic webhook signing

## API Surface

### Endpoints

The first release should implement:

- `POST /runs/crons`
- `POST /threads/{thread_id}/runs/crons`
- `POST /runs/crons/search`
- `POST /runs/crons/count`
- `GET /runs/crons/{cron_id}`
- `PATCH /runs/crons/{cron_id}`
- `DELETE /runs/crons/{cron_id}`

The route shapes should stay LangSmith-style even where implementation is intentionally narrower.

### Request model

The persisted cron record should support:

- `assistant_id`
- cron payload or input body
- optional `metadata`
- optional config/context kwargs needed to create a run
- optional `webhook`
- `schedule` as an RRULE string
- optional `timezone`
- `enabled`

Thread-bound crons also bind a fixed `thread_id`.

Stateless crons do not accept a fixed `thread_id`; each fire creates a new thread before creating a run.

### Response model

Each cron response should expose enough state for clients to reason about scheduling:

- `cron_id`
- `assistant_id`
- optional `thread_id`
- `schedule`
- `timezone`
- `enabled`
- `next_run_at`
- `last_run_at`
- `last_tick_status`
- `last_error`
- `webhook`
- timestamps

Search and count should operate on persisted cron rows, not on run history.

## Scheduler Architecture

### Dedicated process

Cron scheduling should run in a dedicated `agentseek-api scheduler` process.

This is preferred over embedding the scheduler in API servers or workers because:

- API servers stay focused on control-plane HTTP
- workers stay focused on run execution
- cron liveness is explicit in deployment topology
- the scheduler can reuse the existing Redis leader-lock pattern without claiming distributed-runtime parity

### Leader election

Multiple scheduler processes may be started, but only one should be active at a time.

The scheduler should:

- acquire a Redis leader lease using a scheduler-specific key
- renew that lease periodically
- stop dispatching if the lease is lost

This matches the repo's existing Redis worker-lock model and is sufficient for first-release workable parity.

### Polling model

The scheduler loop should:

1. poll for due cron rows where `enabled = true` and `next_run_at <= now`
2. claim due work in small batches
3. create a `CronTick` record for that scheduled occurrence
4. dispatch work according to cron type
5. compute and persist the next valid occurrence from the RRULE
6. persist final tick and webhook-delivery state

The next occurrence should be persisted as part of dispatch handling so a leader failover does not re-emit the same tick indefinitely.

## Persistence Model

### CronJob

Add a `CronJob` table with fields for:

- `cron_id`
- `user_id`
- optional `thread_id`
- `assistant_id`
- `input_json`
- `metadata_json`
- `kwargs_json`
- `schedule_rrule`
- `timezone`
- `enabled`
- `webhook`
- `next_run_at`
- `last_run_at`
- `last_tick_status`
- `last_error`
- `max_webhook_attempts`
- `created_at`
- `updated_at`

### CronTick

Add a `CronTick` table with one row per due occurrence:

- `tick_id`
- `cron_id`
- optional `run_id`
- `scheduled_for`
- `started_at`
- `finished_at`
- `status`
- `skip_reason`
- `error`
- `created_at`

Status values should include:

- `queued`
- `started`
- `success`
- `error`
- `skipped`

### CronWebhookAttempt

Add a `CronWebhookAttempt` table with one row per outbound callback attempt:

- `attempt_id`
- `tick_id`
- `attempt_number`
- `status_code`
- `response_body` or truncated response metadata
- `error`
- `delivered_at`
- `created_at`

This gives the API and operational logs a durable record of delivery behavior without conflating webhook state with run state.

## Dispatch Semantics

### Stateless cron fire

When a stateless cron is due:

1. create a new thread for the authenticated user
2. create and submit a run on that thread using the stored assistant and payload
3. store the resulting `thread_id` and `run_id` on the tick

Each fire is isolated from previous ones.

### Thread-bound cron fire

When a thread-bound cron is due:

1. check whether the target thread already has an active run
2. if idle, submit the stored run payload to that thread
3. if busy, record a `CronTick(status="skipped", skip_reason="thread_busy")`
4. advance the cron to the next RRULE occurrence

First-release policy is explicit skip-on-conflict, not backlog or disable-on-conflict.

This preserves the existing one-active-run-per-thread invariant and keeps behavior observable.

## Webhook Semantics

### Delivery contract

Webhook delivery is best-effort and asynchronous.

For each completed tick:

- build a completion payload containing at least `cron_id`, `thread_id`, `run_id`, `scheduled_for`, `status`, and any error text
- attempt an HTTP `POST` to the configured webhook URL
- retry with bounded backoff up to `max_webhook_attempts`
- persist each attempt and the final delivery outcome

Webhook failure must not change the underlying run result.

### Non-goals

First release does not promise:

- exactly-once webhook delivery
- idempotency tokens
- HMAC signing
- user-configurable retry policies beyond the baked-in bounded policy

Those can be added later without changing the core cron resource model.

## Validation Rules

### Schedule validation

The API should accept RRULE strings, but parse and validate them explicitly.

Policy:

- supported clauses are accepted
- unsupported clauses are rejected with a clear 4xx error
- malformed RRULE strings are rejected with a clear 4xx error
- `timezone` must be a valid IANA timezone if present

The public contract stays RRULE-shaped even if first-release support is partial.

### Resource validation

- `assistant_id` must exist
- thread-bound crons require an accessible thread
- stateless crons must not include a fixed `thread_id`
- `webhook` must be an absolute `http` or `https` URL
- `PATCH` recomputes `next_run_at` whenever schedule, timezone, or enabled state changes

## CLI and Runtime Surface

Add a new CLI subcommand:

- `agentseek-api scheduler`

Runtime roles become:

- `serve`: API only
- `worker`: run execution only
- `scheduler`: cron dispatch and webhook retry loop

Docs should state that cron-capable deployments require:

- API server
- worker
- scheduler

Single-host development convenience can be added later, but should not be required for first-release correctness.

## Error Handling

Expected first-release behavior:

- unsupported RRULE clauses return 4xx with explicit error text
- inaccessible threads or assistants return 404
- thread-bound fire on busy thread records a skipped tick rather than raising to clients
- scheduler lease loss stops dispatch until another scheduler becomes leader
- webhook failure is recorded separately and does not alter run terminal state

## Testing Strategy

### Unit tests

Add failing tests first for:

- RRULE parsing and unsupported-clause rejection
- next-occurrence computation
- thread-bound skip-on-busy behavior
- webhook retry counting and stop conditions

### Integration tests

Add coverage for:

- cron CRUD/search/count
- stateless cron fire creates a new thread and run
- thread-bound cron fire skips when the thread is busy
- only one scheduler leader claims due crons
- webhook attempts are persisted on success and failure

### System tests

Verification should include:

- `/info` reporting `crons: true`
- removal of `crons` from unsupported features
- end-to-end proof that a due cron produces a real run through the existing executor path

## Implementation Split

### PR 1: Cron persistence and API surface

Scope:

- add ORM models
- add cron API models and routes
- add CRUD/search/count behavior
- keep cron feature flag off until scheduler execution lands

Acceptance criteria:

- cron rows persist correctly
- both creation modes are test-covered
- validation is explicit and stable

### PR 2: Scheduler process and due-run dispatch

Scope:

- add scheduler process and Redis leader lease
- claim due crons
- create ticks
- dispatch stateless and thread-bound runs
- implement skip-on-busy semantics

Acceptance criteria:

- only one scheduler dispatches a due cron tick
- stateless and thread-bound crons both create observable tick records
- busy thread-bound ticks are skipped, not duplicated

### PR 3: Webhook retries, docs, and release surface

Scope:

- add webhook delivery attempts and bounded retries
- update `/info`, README, and deployment docs
- add final end-to-end cron verification

Acceptance criteria:

- webhook delivery attempts are durable and test-covered
- public docs describe cron support truthfully
- release claims remain “workable OSS parity,” not full platform parity
