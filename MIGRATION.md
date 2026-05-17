# AgentSeek API Roadmap

> **For agentic workers:** Execute task-by-task with checkbox tracking. Do
> not skip verification commands. Milestones run top-to-bottom.

**Tech stack:** FastAPI, LangGraph >= 1.0, langchain-core >= 1.0,
SQLAlchemy async, aiomysql/asyncpg, pymysql, pytest.

---

## File Map (Primary Surfaces)

- `src/agentseek_api/api/*`: Agent Protocol endpoints.
- `src/agentseek_api/services/*`: run prep/execution and graph wiring.
- `src/agentseek_api/services/sample_graphs.py`: `graph_id` → adapter registry.
- `src/agentseek_api/core/database.py`: metadata + checkpoint lifecycle.
- `src/agentseek_api/core/oceanbase_checkpointer.py`: checkpoint persistence.
- `src/agentseek_api/core/auth_middleware.py`: noop/custom auth loading.
- `src/agentseek_api/settings.py`: env-driven backend configuration.
- `tests/unit/*`, `tests/integration/*`, `tests/e2e/*`: verification layers.
- `examples/graphs/*`: sample LangGraph apps registered with the API.
- `examples/run_sample_graphs.py`: in-process sample proof runner.
- `scripts/test-seekdb.sh`, `scripts/seekdb_embed_launcher.py`: real backend smoke.
- `README.md`, `examples/README.md`, `.env.example`: user-facing docs/config.

---

# Milestone 1 — Core Runtime on SeekDB ✅ (2026-05-13)

Agent Protocol surface with OceanBase/SeekDB-first persistence, delivered
across six phases. All checkboxes below are satisfied.

## Phase 1: Runtime Parity (Assistants/Threads/Runs)

- [x] Validate CRUD + wait/stream semantics.
- [x] Confirm user-scoped access checks on thread/run endpoints.
- [x] Ensure stateless run endpoint mirrors expected request/response contract.
- [x] Verify run lifecycle transitions and error propagation are deterministic.

**Verify** — `uv run pytest tests/unit/test_run_preparation.py tests/unit/test_run_executor.py tests/integration/test_assistants_crud.py tests/integration/test_threads_crud.py tests/integration/test_runs_crud.py -q`

## Phase 2: SeekDB/OceanBase-First Persistence

- [x] Enforce SeekDB-first defaults in settings and docs.
- [x] Confirm metadata DB URL resolution + driver normalization (`mysql+aiomysql`, `postgresql+asyncpg`).
- [x] Ensure checkpointer setup runs once during lifecycle startup.
- [x] Keep async checkpoint mode blocked with a clear runtime guard message.
- [x] Depend on published `langchain-oceanbase==0.4.0` for the LangGraph mysql-family checkpoint backend.

**Verify** — `uv run pytest tests/unit/test_database_manager.py tests/unit/test_oceanbase_checkpointer.py tests/integration/test_metadata_db_config.py -q`

## Phase 3: Auth Baseline and Safety

- [x] `AUTH_TYPE` explicit (`noop` or `custom`) with strict config validation.
- [x] Default identity behavior in noop mode.
- [x] Custom backend import failures surface actionable errors.
- [x] Cross-user data isolation for runs/threads.

**Verify** — `uv run pytest tests/unit/test_auth.py tests/unit/test_auth_deps.py tests/integration/test_auth_scope_matrix.py -q`

## Phase 4: Sample Graph Apps

- [x] Sample graph apps under `examples/graphs/<name>/` (not inside the test tree).
- [x] In-process runner exercises every sample.
- [x] E2E flow scripts remain in `tests/e2e/`.

**Verify** — `uv run python examples/run_sample_graphs.py && uv run python tests/e2e/e2e_inprocess_flow.py`

## Phase 5: Real API + Real Backend Evidence

- [x] Real backend smoke with SeekDB/OceanBase (embed or Docker).
- [x] Live HTTP E2E flow through started server.
- [x] `test-seekdb.sh` runs both `e2e_live_http_flow.py` and `e2e_live_http_multi_graph.py`.

**Verify** — `make test-seekdb && make test-e2e`

## Phase 6: Quality Gate

- [x] Unit + integration suites green.
- [x] Coverage ≥ 90%.
- [x] Docs aligned with behavior/config paths.

**Verify** — `make test && make test-cov`

---

## Milestone 1 — Completion Record (2026-05-13)

- `make test`: **89 passed**.
- `make test-cov`: **91.54% coverage** (gate 90%).
- `make test-e2e`: 2 passed against embedded SeekDB.
- `make test-seekdb`: direct checkpoint smoke + live HTTP flow + multi-graph live HTTP — all green.
- Embedded SeekDB launcher (`scripts/seekdb_embed_launcher.py`) wraps `pylibseekdb.open_with_service`; the Docker path still works unchanged.
- `pyproject.toml` now depends on published `langchain-oceanbase==0.4.0`; the earlier git-source hatchling workaround is no longer needed.

### Sample Graph Apps (shipped in Milestone 1)

Four offline-runnable LangGraph apps ship under `examples/graphs/` and are
registered with the API through `src/agentseek_api/services/sample_graphs.py`:

| graph_id              | what it demonstrates                                      |
| --------------------- | --------------------------------------------------------- |
| `stress_test`         | deterministic async loop with configurable delay          |
| `subgraph_agent`      | outer router delegating to a compiled inner subgraph      |
| `react_agent`         | tool-calling ReAct loop (scripted `call_model`, offline)  |
| `subgraph_hitl_agent` | nested subgraph + `interrupt()` human-in-the-loop pattern |

`LangGraphService` maintains a registry keyed by `graph_id` with
`prepare_input`/`extract_output` adapters per entry. `run_preparation` reads
`assistant.graph_id` and passes it to `run_executor`, which calls
`graph.ainvoke` on the matching graph and stores the adapted output plus
the SeekDB/OceanBase checkpoint.

---

# Milestone 2 — Developer-Defined Graphs + HITL Resume 🚧

**Goal:** A developer can plug their own LangGraph app into an AgentSeek API
deployment without editing the server source, and can interrupt/resume
runs through the HTTP surface.

## Phase 2.1: External Graph Registration

- [ ] Introduce a config path (env var `AGENTSEEK_GRAPHS=path/to/manifest.json`, or equivalent TOML) that lists `graph_id → "module.path:symbol"` entries.
- [ ] `LangGraphService` loads the manifest at startup; entries merge with the bundled sample registry, with user entries winning on conflict.
- [ ] Adapter selection: manifest entries may specify optional `prepare_input`/`extract_output` dotted paths; otherwise fall back to the `messages`-style adapters already in `sample_graphs.py`.
- [ ] Document the manifest schema in `examples/README.md` plus a minimal "hello world" example outside `examples/graphs/` (e.g., `examples/external_graph/`).
- [ ] Unit tests: manifest parsing, import errors surfaced cleanly, user-vs-bundled precedence.

**Verify**
- `uv run pytest tests/unit/test_sample_graphs.py tests/unit/test_langgraph_service.py tests/unit/test_graph_manifest.py -q`
- `uv run python examples/external_graph/run.py` (new script that registers via manifest + proves an in-process invoke).

**Exit criteria** — A dev can register a graph by setting one env var and restarting the server.

## Phase 2.2: Interrupt / Resume Through the API

- [ ] Compile every registered graph with a LangGraph checkpointer (`InMemorySaver` in tests, a persistent one — likely SQLite/OceanBase-backed — in production). Keep the existing `OceanBaseCheckpointSaver` row for audit.
- [ ] When a run interrupts, persist run status as `interrupted` instead of `success`, expose the interrupt payload on `RunRead`.
- [ ] New endpoint `POST /threads/{thread_id}/runs/{run_id}/resume` accepts `{"resume": <value>}` and calls `graph.ainvoke(Command(resume=value), config=...)` with the original thread/run ids.
- [ ] Integration tests cover: interrupt returned on first run → resume continues to completion.
- [ ] E2E `tests/e2e/e2e_live_http_resume_flow.py` drives `subgraph_hitl_agent` through interrupt → resume.

**Verify**
- `uv run pytest tests/integration/test_runs_resume.py -q`
- `make test-seekdb` (updated to also drive the resume E2E).

**Exit criteria** — `subgraph_hitl_agent` can be interrupted and resumed end-to-end through the HTTP API against a real SeekDB.

## Phase 2.3: Token-Level Streaming

- [x] Upgrade `GET /threads/{thread_id}/runs/{run_id}/stream` to emit intermediate events from `graph.astream_events` (node starts, tool calls, message tokens) alongside the existing `start`/`end` markers.
- [x] Backward compatibility: the current start/end contract still works for consumers that don't care about sub-events.
- [x] Integration tests: assert that a `react_agent` run yields at least one tool call event and one message-chunk event.

**Verify** — `uv run pytest tests/integration/test_runs_streaming.py tests/integration/test_runs_streaming_errors.py -q`

**Exit criteria** — Streamed runs expose enough granularity for a UI to render tool calls and token-by-token replies.

## Phase 2.4: Quality Gate

- [ ] Unit + integration suites green.
- [ ] Coverage ≥ 90%.
- [ ] `make test-seekdb` green (now covering resume + streaming E2E).
- [ ] README + `examples/README.md` updated with the manifest, resume, and streaming docs.

---

# Milestone 3 — First-Class CLI Surface (next sprint)

**Goal:** Replace the current mix of raw `uvicorn` and `make` invocations with
an `agentseek` developer CLI that is a **strict superset** of the current
LangGraph CLI surface. `agentseek` must accept `langgraph.json` directly,
support the same top-level commands and core options, and then add
AgentSeek-specific extensions such as `serve` and any runtime-specific helpers.
If parity is not implemented for a LangGraph CLI command, the milestone is not
done.

**Reference shape**
- Dedicated CLI package plus console-script entrypoint (`[project.scripts]`).
- Config auto-discovery (`agentseek.json`, then `langgraph.json`) instead of
  forcing every command to take explicit paths. `langgraph.json` compatibility
  is a hard requirement, not a best-effort fallback.
- LangGraph CLI command parity first: `dev`, `build`, `deploy`, `up`,
  `dockerfile`.
- AgentSeek extensions second: `serve`, `version`, and any runtime-specific
  commands that do not break LangGraph CLI compatibility.
- Shared option handling for `--host`, `--port`, `--config`, and `--env-file`.

## Current Status (2026-05-16)

- Implemented and locally verified: `version`, `serve`, `dev`, `build`, `up`,
  `dockerfile`, config auto-discovery (`agentseek.json` then `langgraph.json`),
  and embeddable module entrypoints (`register_subcommands`, `run_namespace`).
- `deploy` parsing/help surface exists, but the command body remains
  intentionally unimplemented in the current milestone slice.
- GitHub CI now covers the CLI on Linux/macOS/Windows plus Docker-backed
  runtime paths through `CLI Compatibility`, `CLI Docker Runtime`, and
  `CLI Dev Sample Graphs`.
- Docker CI specifically verifies built-image startup, container health,
  manifest-driven sample graphs, and custom auth inside the container runtime.

## Phase 3.1: CLI Packaging + Entry Point

- [x] Add a small `agentseek_cli` package (or equivalent module under
  `src/agentseek_api/cli.py`) and expose `agentseek` via `[project.scripts]`.
- [x] Keep the first implementation thin: orchestrate the existing FastAPI app,
  current env/config loader, and existing test/build commands instead of
  duplicating runtime logic.
- [x] Add `agentseek version` that reports the CLI/package version and the
  installed `agentseek-api` version.
- [x] Define command-compatibility policy explicitly: when `agentseek` and
  LangGraph CLI overlap, `agentseek` should accept the LangGraph CLI command
  names and core options unchanged.

## Phase 3.2: Core Runtime Commands

- [x] `agentseek dev`: wraps `uvicorn agentseek_api.main:app --reload`, with
  LangGraph CLI-compatible flags including `-c/--config`, `--host`, `--port`,
  `--no-reload`, `--n-jobs-per-worker`, `--debug-port`, `--wait-for-client`,
  `--no-browser`, `--studio-url`, `--allow-blocking`, and `--tunnel`.
- [x] `agentseek build`: LangGraph CLI-compatible image build surface with
  `--platform`, `-t/--tag`, `--pull/--no-pull`, and `-c/--config`.
- [ ] `agentseek deploy`: LangGraph CLI-compatible deploy surface, including
  inherited build flags plus `--api-key`, `--name`, `--deployment-id`,
  `--deployment-type`, `--no-wait`, `--verbose`, and subcommands
  `deploy list`, `deploy revisions list`, `deploy delete`, and `deploy logs`.
- [x] `agentseek up`: LangGraph CLI-compatible local Docker runtime surface
  with `--wait`, `--base-image`, `--image`, `--postgres-uri`, `--watch`,
  `--debugger-base-url`, `--debugger-port`, `--verbose`, `-c/--config`,
  `-d/--docker-compose`, `-p/--port`, `--pull/--no-pull`, and
  `--recreate/--no-recreate`.
- [x] `agentseek dockerfile`: LangGraph CLI-compatible Dockerfile generation
  surface with `-c/--config`.
- [x] `agentseek serve`: same runtime surface without reload, intended for
  container entrypoints and smoke environments. This is an AgentSeek extension
  and does not replace `up`.
- [x] Config discovery should accept `agentseek.json` first, then full
  `langgraph.json` layouts without requiring users to rewrite the file into an
  AgentSeek-only shape.

## Phase 3.3: Build / Docker Integration

- [x] `agentseek dockerfile [SAVE_PATH]`: generate a Dockerfile from the active
  config, preserving LangGraph CLI expectations while layering in
  AgentSeek-specific runtime pieces only where needed.
- [x] `agentseek build -t <tag>`: build the container image from the generated
  Dockerfile or an equivalent in-memory template without dropping LangGraph CLI
  flags/behavior.
- [x] `agentseek up` should use the generated/built image path and preserve the
  LangGraph CLI local-Docker workflow instead of deferring that command out of
  scope.
- [x] Config parsing handles the runtime-critical LangGraph CLI keys used by
  the current CLI slice, including `dependencies`, `graphs`, `env`, `auth`,
  image/runtime build settings, and manifest-based Docker generation.
  Endpoint-level keys such as `store`, `http`, and `api_version` are currently
  tolerated/ignored in the CLI layer and remain deferred at the runtime/API
  layer.

## Phase 3.4: Tests + Docs

- [x] Unit tests for CLI parsing, config discovery precedence, and the command
  lines emitted for `dev`, `build`, `deploy`, `up`, `dockerfile`, and `serve`.
- [ ] Update `README.md` quickstart to prefer `agentseek dev` once available,
  while keeping the raw `uvicorn` command documented as the low-level fallback.
- [ ] Add a minimal `agentseek.json` example alongside the existing
  `langgraph.json`-compatible graph mapping examples.
- [ ] Add a fixture test proving the basic LangGraph config example runs
  unchanged under `agentseek`:
  ```json
  {
    "$schema": "https://langgra.ph/schema.json",
    "dependencies": ["."],
    "graphs": {
      "chat": "chat.graph:graph"
    }
  }
  ```

**Verify**
- `uv run pytest tests/unit/test_cli.py -q`
- `uv run agentseek version`
- `uv run agentseek dev --help`
- `uv run agentseek deploy --help`
- `uv run agentseek up --help`
- `uv run agentseek build --help`
- `uv run agentseek dockerfile --help`
- `uv run agentseek deploy list --help`

**Exit criteria** — A developer can install the package, point `agentseek` at a
valid `langgraph.json`, and use the full LangGraph CLI command surface through
`agentseek` without losing command/option compatibility. AgentSeek-only
commands are additive on top of that baseline.

---

# Milestone 4 — Assistant Configuration + Auth Samples (proposed)

**Goal:** Real-world shaped multi-tenant scenarios: per-assistant
configuration flowed into the graph, first-class JWT auth sample, custom
HTTP routes mounted on the same app.

- [ ] `Assistant` gains a `config` JSON column; `RunnableConfig` threads it into `graph.ainvoke` so graphs can branch on per-assistant settings (prompts, tool lists, model selection).
- [ ] JWT auth sample backend under `examples/auth/jwt_backend.py`, loadable via `AUTH_TYPE=custom AUTH_MODULE_PATH=examples.auth.jwt_backend:backend`.
- [ ] Custom routes sample under `examples/routes/custom_routes.py` showing how a dev mounts extra FastAPI routes that share auth + settings.
- [ ] E2E proves: create assistant with config → run → graph observes config; authenticated request with JWT → run succeeds; unauthenticated → 401.

---

# Milestone 5 — LangSmith Agent Server API Compatibility

**Goal:** Close the largest gaps between AgentSeek's current HTTP surface and
the LangSmith Agent Server API so LangSmith-style clients can use this server
with minimal or no adaptation.

## Current Status (2026-05-17)

- Milestone 5's core assistants / threads / runs / stateless-runs
  compatibility slice is now implemented and verified.
- The mysql-family checkpoint backend is now pinned to published
  `langchain-oceanbase==0.4.0` rather than a git dependency.
- The compatibility layer is CI-verified across:
  - MySQL-family checkpoint validation on `mysql`, `seekdb`, and `oceanbase`
  - PostgreSQL metadata + checkpoint validation on `mysql`, `seekdb`, and
    `oceanbase`
  - Docker runtime smoke through `CLI Docker Runtime`
  - Manifest-driven live sample graphs through `CLI Dev Sample Graphs`
- Latest green proof run: GitHub Actions run `25984792293`, including:
  - `CLI Docker Runtime`
  - `CLI Dev Sample Graphs`
  - `MySQL-Family Checkpoint Validation` (`mysql`, `seekdb`, `oceanbase`)
  - `PostgreSQL Metadata + Checkpoint Validation`
    (`mysql`, `seekdb`, `oceanbase`)
- Previously deferred `POST /threads/{thread_id}/copy`,
  `POST /threads/prune`, and `POST /runs/batch` are now shipped and no longer
  tracked as deferred scope.
- Locally re-verified on this branch with:
  - `uv run pytest -q` → `212 passed, 7 skipped`
  - `uv run ruff check src tests`
  - `uv run python -m py_compile $(rg --files src tests scripts examples -g '*.py')`
  - `SEEKDB_MODE=embed bash ./scripts/test-checkpoints.sh`

**Implemented compatibility surface**

- System:
  - `GET /health`
  - `GET /ok`
  - `GET /info`
  - `GET /metrics`
- Assistants:
  - `POST /assistants`
  - `GET /assistants`
  - `POST /assistants/search`
  - `POST /assistants/count`
  - `GET /assistants/{assistant_id}`
  - `PATCH /assistants/{assistant_id}`
  - `DELETE /assistants/{assistant_id}`
  - `GET /assistants/{assistant_id}/graph`
  - `GET /assistants/{assistant_id}/schemas`
  - `GET /assistants/{assistant_id}/subgraphs`
  - `GET /assistants/{assistant_id}/subgraphs/{namespace}`
  - `POST /assistants/{assistant_id}/versions`
  - `POST /assistants/{assistant_id}/latest`
- Threads:
  - `POST /threads`
  - `GET /threads`
  - `POST /threads/search`
  - `POST /threads/count`
  - `GET /threads/{thread_id}`
  - `PATCH /threads/{thread_id}`
  - `DELETE /threads/{thread_id}`
  - `POST /threads/{thread_id}/copy`
  - `POST /threads/prune`
  - `GET /threads/{thread_id}/state`
  - `GET /threads/{thread_id}/history`
  - `POST /threads/{thread_id}/history`
  - `GET /threads/{thread_id}/state/{checkpoint_id}`
  - `POST /threads/{thread_id}/state`
  - `POST /threads/{thread_id}/state/checkpoint`
  - `GET /threads/{thread_id}/stream`
  - `POST /threads/{thread_id}/commands`
  - `POST /threads/{thread_id}/stream/events`
- Thread runs:
  - `POST /threads/{thread_id}/runs`
  - `GET /threads/{thread_id}/runs`
  - `GET /threads/{thread_id}/runs/{run_id}`
  - `GET /threads/{thread_id}/runs/{run_id}/wait`
  - `POST /threads/{thread_id}/runs/wait`
  - `POST /threads/{thread_id}/runs/stream`
  - `GET /threads/{thread_id}/runs/{run_id}/stream`
  - `POST /threads/{thread_id}/runs/{run_id}/resume`
  - `POST /threads/{thread_id}/runs/{run_id}/cancel`
  - `GET /threads/{thread_id}/runs/{run_id}/join`
  - `DELETE /threads/{thread_id}/runs/{run_id}`
- Stateless runs:
  - `POST /runs`
  - `POST /runs/wait`
  - `POST /runs/stream`
  - `POST /runs/batch`
  - `POST /runs/cancel`

**Notable behavioral work completed**

- Assistant create / patch now persist and merge LangSmith-relevant
  `metadata`, `config`, `context`, and `description` fields.
- Thread create / patch now persist `config`, reject unsupported patch fields
  with `422`, and merge metadata instead of replacing it.
- Thread state / history / checkpoint routes now read from the LangGraph
  checkpointer rather than reconstructing state from `Run` rows.
- Cancelled runs no longer race back to `success` in persisted run rows.
- Cancelled-run checkpoints are filtered out of thread state/history lookup so
  cancelled runs do not leak completed state through thread endpoints.
- Empty-thread synthetic checkpoint ids are now resolvable through
  `/threads/{thread_id}/state/{checkpoint_id}` and
  `/threads/{thread_id}/state/checkpoint`.
- Fallback checkpoint copy logic is now parent-order safe and does not depend
  on saver iteration order.
- `/info` now advertises `protocol_v2: true` to match the mounted protocol-v2
  routes.

## Phase 5.1: Spec-Native Surface for Existing Core Flows

- [x] Add `GET /ok` alongside the existing health endpoint, returning the
  LangSmith-style `{ "ok": true }` shape.
- [x] Add `GET /metrics` with at least the documented `prometheus|json` output
  switch, even if the first cut is minimal runtime/process metrics.
- [x] Keep existing legacy endpoints (`GET /assistants`, `GET /threads`,
  `GET /threads/{thread_id}/runs/{run_id}/wait`, etc.) for backward
  compatibility, but add the spec-native route variants first.
- [x] Add `POST /assistants/search` and `POST /threads/search` so list/search
  follows the LangSmith contract instead of custom `GET` list-only behavior.
- [x] Add `POST /threads/{thread_id}/runs/wait` and
  `POST /threads/{thread_id}/runs/stream` so stateful run creation matches the
  documented background/wait/stream split.
- [x] Add `POST /runs/wait` and `POST /runs/stream` so stateless runs expose
  the same wait/stream variants as LangSmith deployments.

**Verify**
- `uv run pytest tests/integration/test_system_endpoints.py tests/integration/test_assistants_compat.py tests/integration/test_threads_compat.py tests/integration/test_runs_compat.py -q`

**Exit criteria** — A LangSmith-oriented client sees the expected first-wave
paths for health, list/search, and run creation modes without relying on
AgentSeek-specific aliases.

## Phase 5.2: Schema Expansion and Response Parity

- [x] Expand `AssistantCreate` / `AssistantRead` toward the LangSmith contract:
  `assistant_id`, `graph_id`, `config`, `context`, `metadata`, `name`,
  `description`, `created_at`, `updated_at`, and version tracking fields.
- [~] Expand `ThreadCreate` / `ThreadRead` toward the LangSmith contract:
  caller-provided `thread_id`, `if_exists`, `ttl`, and `supersteps` remain
  deferred, but response fields for `updated_at`, `state_updated_at`, `config`,
  and `status` are now implemented.
- [~] Expand run payloads to accept the higher-value LangSmith fields first:
  `metadata`, `config`, `context`, and `multitask_strategy` are now supported;
  `command`, `stream_mode`, `stream_resumable`, and `if_not_exists` remain
  deferred.
- [x] Expand `RunRead` toward LangSmith's run resource shape:
  `created_at`, `updated_at`, `metadata`, `kwargs`, and
  `multitask_strategy`, while preserving AgentSeek's useful `output`,
  `interrupts`, and `last_error` fields where they do not conflict.
- [x] Align `GET /info` with the documented server info payload:
  `version`, `langgraph_py_version`, `flags`, and `metadata`.

**Verify**
- `uv run pytest tests/integration/test_system_endpoints.py tests/integration/test_assistants_compat.py tests/integration/test_langsmith_compat_coverage.py tests/integration/test_langsmith_compat_extra.py -q`

**Exit criteria** — Shared routes no longer just look similar; their request
and response bodies are close enough to satisfy LangSmith client expectations.

## Phase 5.3: Thread State and History on Top of Existing Checkpoints

- [x] Add `GET /threads/{thread_id}/state`.
- [x] Add `GET /threads/{thread_id}/history` and `POST /threads/{thread_id}/history`.
- [x] Add `POST /threads/{thread_id}/state` and
  `POST /threads/{thread_id}/state/checkpoint` on top of the current LangGraph
  checkpointer wiring.
- [x] Add thread patch/delete/copy/prune primitives:
  `PATCH /threads/{thread_id}`, `DELETE /threads/{thread_id}`,
  `POST /threads/{thread_id}/copy`, `POST /threads/prune`.
- [x] Ensure thread status (`idle`, `busy`, `interrupted`, `error`) is derived
  deterministically from persisted run/checkpoint state rather than inferred
  ad hoc per request.

**Verify**
- `uv run pytest tests/integration/test_threads_compat.py tests/integration/test_langsmith_compat_coverage.py tests/integration/test_langsmith_compat_extra.py tests/unit/test_thread_routes_unit.py tests/unit/test_thread_checkpoint_store.py -q`

**Exit criteria** — The server exposes the checkpoint-backed thread state that
LangSmith clients expect, not just metadata rows plus run records.

## Phase 5.4: Streaming, Resume, and Cancellation Parity

- [x] Keep the current SSE `message_chunk` proof path intact while adding the
  documented request paths and query/header semantics.
- [x] Add `POST /threads/{thread_id}/runs/{run_id}/cancel` and
  `POST /runs/cancel`.
- [ ] Support resumable run streaming semantics where practical:
  `stream_resumable`, `Last-Event-ID`, and explicit persisted event replay
  instead of the current in-memory-only broker.
- [~] Evaluate whether the existing custom
  `POST /threads/{thread_id}/runs/{run_id}/resume` should remain as a
  compatibility extension, or whether resume should be represented primarily
  through LangSmith protocol/state commands.
  Current status: kept as a compatibility extension for now.
- [x] Add the protocol-v2 surfaces:
  `POST /threads/{thread_id}/commands` and
  `POST /threads/{thread_id}/stream/events`.

**Verify**
- `uv run pytest tests/integration/test_runs_compat.py tests/integration/test_run_cancellation_async.py tests/e2e/test_langsmith_compat_live.py -q`
- `SEEKDB_MODE=embed bash ./scripts/test-checkpoints.sh`

**Exit criteria** — Streaming is compatible at both the basic run SSE layer
and the newer protocol-v2 command/event layer, with documented remaining gaps
limited to restart-safe replay and full resumable event semantics.

## Phase 5.5: Auth and Remaining High-Value Surface

- [ ] Add a first-class `X-Api-Key` auth mode suitable for LangSmith-style
  deployments instead of requiring a custom backend for that contract.
- [x] Preserve the existing `noop` and `custom` modes for local dev and custom
  platform integrations.
- [x] Add assistants patch/delete/count and the highest-value assistant
  introspection endpoints (`graph`, `schemas`, `subgraphs`) where the data is
  derivable from the registered graph registry.
- [x] Defer Store, Crons, MCP, and A2A until the assistants/threads/runs
  compatibility work above is stable, but keep them tracked as explicit
  follow-on milestones rather than silent gaps.

**Verify**
- `uv run pytest tests/unit/test_auth.py tests/unit/test_auth_deps.py tests/unit/test_assistant_routes_unit.py -q`
- GitHub Actions: `CLI Docker Runtime`, `CLI Dev Sample Graphs`,
  `MySQL-Family Checkpoint Validation`,
  `PostgreSQL Metadata + Checkpoint Validation`

**Exit criteria** — Core LangSmith deployment auth and assistant lifecycle
operations are covered well enough that the remaining gaps are mostly advanced
surfaces, not first-contact blockers.

**Remaining gaps after Milestone 5**

- `X-Api-Key` auth for LangSmith-hosted deployment semantics.
- Full Store API parity.
- Crons / scheduler.
- MCP and A2A endpoint parity.
- Restart-safe persisted streaming event replay (`Last-Event-ID`,
  resumable event logs, durable token/event history).
- Additional LangSmith request fields such as caller-provided `thread_id`,
  `ttl`, `if_exists`, `supersteps`, `command`, `stream_mode`,
  `stream_resumable`, and `if_not_exists`.

---

# Deferred Scope (Explicitly Out of Milestone 1, 2, and 5)

- Crons / scheduler.
- MCP and A2A endpoint parity.
- Full Store API parity.
- Distributed worker topology and lease/reaper production architecture.
