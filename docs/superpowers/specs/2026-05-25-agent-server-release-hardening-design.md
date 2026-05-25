# AgentSeek API Agent Server Release Hardening Design

## Goal

Ship a workable OSS parity release for the LangSmith Agent Server core surface without claiming full platform parity.

This work is explicitly scoped to:

- hardening runtime correctness for threads, runs, and streaming
- removing misleading compatibility behavior where the API currently returns placeholder success
- aligning public release claims with the implementation that actually exists

This work is explicitly out of scope:

- cron jobs and scheduling
- distributed runtime and multi-worker orchestration parity
- full assistant version-management parity
- large-scale query/index optimization beyond obvious correctness issues

## Release Standard

The release should be safe to present as:

- workable OSS parity for assistants, threads, runs, streaming, Store API, MCP, and A2A
- not full LangSmith Agent Server parity

The release should not imply:

- cron support
- multi-worker scale-out semantics similar to LangSmith Deployment
- complete helper endpoint behavior where only compatibility stubs exist today

## Runtime Policy

### Thread admission

AgentSeek API will enforce single active run per thread.

Policy:

- if a thread already has a non-terminal run, a new thread-scoped run request is rejected with HTTP `409`
- the default `multitask_strategy` remains `enqueue` in request models for wire compatibility, but the server will not silently accept it unless real serialized execution exists
- for this milestone, `enqueue` is treated as unsupported for thread-backed runs and returns the same `409` conflict as other concurrent submissions

Rationale:

- the current code records `multitask_strategy` but does not enforce admission control or true per-thread serialization
- rejecting conflicting runs is safer than pretending queueing exists
- this preserves thread-state and checkpoint integrity

### Stream semantics

`GET /threads/{thread_id}/stream` must behave as a live thread stream, not as a closed replay of runs visible at connection start.

Policy:

- the endpoint replays persisted thread events after the requested cursor
- after replay, it remains attached to live thread events until the client disconnects
- future runs started on the same thread after the stream opens must appear on the same connection
- the endpoint no longer exits immediately after replaying the initial snapshot

`GET /threads/{thread_id}/runs/{run_id}/stream` remains a run-scoped resumable SSE stream backed by persisted event replay.

Policy:

- `Last-Event-ID` replay continues to work for persisted run events
- docs and endpoint behavior should only claim resumability that the implementation actually guarantees
- unsupported query semantics from LangSmith docs, such as `cancel_on_disconnect` or `stream_mode` query handling, are not introduced unless implemented end-to-end

## Compatibility Surface Policy

### Assistant helper endpoints

Compatibility helper endpoints must either expose meaningful data or fail/declare limits clearly.

Required behavior for this milestone:

- `GET /assistants/{assistant_id}/subgraphs` and namespaced variants must not present fabricated parity
- `POST /assistants/{assistant_id}/versions` must describe current semantics truthfully rather than implying full version history management
- `POST /assistants/{assistant_id}/latest` must reflect actual behavior or be limited explicitly
- `DELETE /assistants/{assistant_id}?delete_threads=true` remains unsupported unless thread deletion semantics are implemented safely

Preferred approach:

- keep the endpoints, but narrow responses and docs so they are truthful compatibility helpers rather than fake feature completions
- only implement deeper behavior where the repo already has real underlying state to support it

### Public claims

README, `/info`, and compatibility language must describe the current server as:

- core Agent Server-compatible on the implemented surfaces
- intentionally incomplete on crons, distributed runtime, and certain helper APIs

## PR Split

### PR 1: Runtime semantics hardening

Scope:

- enforce one active run per thread
- reject concurrent thread-backed submissions with `409`
- make thread stream stay live across future runs
- tighten run stream resumability claims and tests
- add integration coverage for admission control and live thread streaming

Files likely touched:

- `src/agentseek_api/services/run_preparation.py`
- `src/agentseek_api/api/threads.py`
- `src/agentseek_api/api/runs.py`
- `tests/integration/test_*stream*`
- `tests/integration/test_*runs*`

Acceptance criteria:

- a second run on the same thread while one is pending/running/interrupted is rejected
- thread stream opened before a later run still receives that later run's events
- existing run replay behavior stays green

### PR 2: Assistant helper surface truthfulness

Scope:

- remove placeholder-success semantics from helper endpoints
- pin actual behavior for subgraphs, versions, latest, and delete_threads handling
- update tests to verify the intended limited contract

Files likely touched:

- `src/agentseek_api/api/assistants.py`
- `tests/integration/test_assistants_*`
- `tests/integration/test_langsmith_compat_*`
- possibly `README.md` for endpoint notes if required by behavior changes

Acceptance criteria:

- helper endpoints no longer imply unavailable server features
- responses are stable and test-covered

### PR 3: Release claim cleanup

Scope:

- document the actual OSS parity boundary
- align README future work and compatibility notes with implemented behavior
- keep the manual provider-streaming workflow as the canonical live proof path
- add a release checklist section for backend matrix plus manual provider-stream verification

Files likely touched:

- `README.md`
- possibly `examples/README.md`
- possibly `AGENTS.md` or release notes if needed

Acceptance criteria:

- a community user reading the docs will not infer full LangSmith parity
- remaining gaps are explicit and intentional

## Data Flow Changes

### Run admission

Before creating a new thread-backed run:

1. load the target thread for the authenticated user
2. query for an existing non-terminal run on that thread
3. if one exists, reject with `409` and a precise conflict message
4. otherwise create the pending run and submit it normally

This check must apply consistently to:

- `POST /threads/{thread_id}/runs`
- `POST /threads/{thread_id}/runs/wait`
- `POST /threads/{thread_id}/runs/stream`
- protocol v2 `run.start`

### Live thread stream

Thread stream behavior will be:

1. replay persisted thread events after the requested cursor
2. merge any in-memory broker snapshot not yet persisted
3. if Redis executor is active, continue polling persisted thread events while the thread remains active and while new runs arrive later
4. if inline executor is active, remain subscribed to the protocol broker for future thread events
5. terminate only when the client disconnects or the server shuts down the response

The key design constraint is that the stream is thread-centric, not a one-time reduction over current run IDs.

## Error Handling

- concurrent thread-backed run submission returns `409`
- the error body must explain that another run is already active for the thread
- helper endpoints that cannot provide real parity should return clear limited responses or explicit errors, not empty success payloads that look complete
- no new silent fallback should be introduced for unsupported concurrency semantics

## Testing Strategy

### PR 1

Add failing integration tests first for:

- second run on same thread returns `409` while first run is active
- protocol `run.start` also returns a conflict on an active thread
- thread stream opened before a later run remains open long enough to receive the later run events
- existing replay-from-cursor behavior still works

### PR 2

Add failing tests first for:

- helper endpoints returning explicit limited semantics instead of placeholder parity
- any changed status codes or response shape

### PR 3

Verification:

- `uv run pytest tests/unit tests/integration -q`
- relevant targeted e2e or CI-backed backend suites where local infra exists
- manual workflow remains unchanged for live provider streaming

## Risks

- changing thread stream from finite replay to live follow may expose brittle assumptions in current tests
- admission control may break tests that currently create back-to-back runs on the same thread without waiting
- if helper endpoints change shape too aggressively, compatibility tests may need coordinated updates

These are acceptable because the current behavior is less safe than the proposed behavior for community release.

## Recommendation

Proceed in three PRs in this order:

1. runtime semantics hardening
2. assistant helper surface truthfulness
3. release claim cleanup

This order closes the correctness gap first, then removes misleading compatibility behavior, then aligns the public release surface to the hardened implementation.
