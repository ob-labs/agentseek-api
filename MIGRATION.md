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
- [x] Depend on upstream `langchain-oceanbase` branch `release/0.4.0` until checkpoint package is published.

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
- `pyproject.toml` carries `[tool.hatch.metadata] allow-direct-references = true` so the git-sourced `langchain-oceanbase` dependency builds cleanly under hatchling.

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

## Phase 3.1: CLI Packaging + Entry Point

- [ ] Add a small `agentseek_cli` package (or equivalent module under
  `src/agentseek_api/cli.py`) and expose `agentseek` via `[project.scripts]`.
- [ ] Keep the first implementation thin: orchestrate the existing FastAPI app,
  current env/config loader, and existing test/build commands instead of
  duplicating runtime logic.
- [ ] Add `agentseek version` that reports the CLI/package version and the
  installed `agentseek-api` version.
- [ ] Define command-compatibility policy explicitly: when `agentseek` and
  LangGraph CLI overlap, `agentseek` should accept the LangGraph CLI command
  names and core options unchanged.

## Phase 3.2: Core Runtime Commands

- [ ] `agentseek dev`: wraps `uvicorn agentseek_api.main:app --reload`, with
  LangGraph CLI-compatible flags including `-c/--config`, `--host`, `--port`,
  `--no-reload`, `--n-jobs-per-worker`, `--debug-port`, `--wait-for-client`,
  `--no-browser`, `--studio-url`, `--allow-blocking`, and `--tunnel`.
- [ ] `agentseek build`: LangGraph CLI-compatible image build surface with
  `--platform`, `-t/--tag`, `--pull/--no-pull`, and `-c/--config`.
- [ ] `agentseek deploy`: LangGraph CLI-compatible deploy surface, including
  inherited build flags plus `--api-key`, `--name`, `--deployment-id`,
  `--deployment-type`, `--no-wait`, `--verbose`, and subcommands
  `deploy list`, `deploy revisions list`, `deploy delete`, and `deploy logs`.
- [ ] `agentseek up`: LangGraph CLI-compatible local Docker runtime surface
  with `--wait`, `--base-image`, `--image`, `--postgres-uri`, `--watch`,
  `--debugger-base-url`, `--debugger-port`, `--verbose`, `-c/--config`,
  `-d/--docker-compose`, `-p/--port`, `--pull/--no-pull`, and
  `--recreate/--no-recreate`.
- [ ] `agentseek dockerfile`: LangGraph CLI-compatible Dockerfile generation
  surface with `-c/--config`.
- [ ] `agentseek serve`: same runtime surface without reload, intended for
  container entrypoints and smoke environments. This is an AgentSeek extension
  and does not replace `up`.
- [ ] Config discovery should accept `agentseek.json` first, then full
  `langgraph.json` layouts without requiring users to rewrite the file into an
  AgentSeek-only shape.

## Phase 3.3: Build / Docker Integration

- [ ] `agentseek dockerfile [SAVE_PATH]`: generate a Dockerfile from the active
  config, preserving LangGraph CLI expectations while layering in
  AgentSeek-specific runtime pieces only where needed.
- [ ] `agentseek build -t <tag>`: build the container image from the generated
  Dockerfile or an equivalent in-memory template without dropping LangGraph CLI
  flags/behavior.
- [ ] `agentseek up` should use the generated/built image path and preserve the
  LangGraph CLI local-Docker workflow instead of deferring that command out of
  scope.
- [ ] Config parsing must handle core LangGraph CLI keys cleanly, including at
  least `dependencies`, `graphs`, `env`, `auth`, `store`, `http`, and
  `api_version`. Unsupported keys must fail clearly instead of being silently
  ignored.

## Phase 3.4: Tests + Docs

- [ ] Unit tests for CLI parsing, config discovery precedence, and the command
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

# Deferred Scope (Explicitly Out of Milestone 1 & 2)

- Crons / scheduler.
- MCP and A2A endpoint parity.
- Thread copy/prune and run-batch APIs.
- Distributed worker topology and lease/reaper production architecture.
