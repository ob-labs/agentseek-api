# AgentSeek API

AgentSeek API is a Python package and CLI for running LangGraph / LangChain
apps behind an Agent Protocol-compatible HTTP API. It ships a FastAPI runtime,
LangGraph graph loading, persistent metadata, and OceanBase / SeekDB checkpoint
support.

Install the package as a library when you want to embed the runtime. Install the
same package as a CLI when you want to run it from a project directory:

```bash
uv sync
uv run agentseek-api dev
```

## Status

The current codebase has already delivered the core runtime, external graph
loading, HITL resume, token streaming, a first-class standalone CLI, and the
main [Agent Protocol](https://github.com/langchain-ai/agent-protocol)
thread / run / streaming foundations.

- Runtime: assistants, threads, thread runs, stateless runs, SeekDB /
  OceanBase-first persistence, noop/custom auth, sample graphs, and real
  backend validation.
- Developer integration: `AGENTSEEK_GRAPHS` manifest loading, external graph
  examples, interrupt/resume, and SSE `message_chunk` streaming.
- CLI: `agentseek-api dev`, `serve`, `build`, `up`, `dockerfile`, and
  `version`, plus parent-CLI embedding through `register_subcommands(...)`.
- Compatibility: thread and run creation, wait, stream, cancel, copy, prune,
  history, protocol-v2 thread event streaming, and server status endpoints.

<details>
<summary>Achieved so far</summary>

### Milestone 1: Core runtime and persistence

- Shipped the core CRUD / wait / stream API surface for assistants, threads,
  thread runs, and stateless runs.
- Locked the mysql-family checkpoint backend to published
  `langchain-oceanbase==0.4.0`.
- Established explicit auth modes: `AUTH_TYPE=noop` and `AUTH_TYPE=custom`.
- Added real backend validation through `make test-seekdb` and `make test-e2e`.
- Cleared the quality gate with 90%+ coverage.

### Milestone 2: Developer-defined graphs, resume, and streaming

- Added manifest-driven graph loading through `agentseek.json`,
  `langgraph.json`, or `AGENTSEEK_GRAPHS`.
- Shipped external graph examples and bundled sample graphs.
- Added `POST /threads/{thread_id}/runs/{run_id}/resume` for HITL resume flows.
- Upgraded streaming to emit intermediate SSE events including `message_chunk`.

### Milestone 3: First-class CLI

- Shipped the standalone `agentseek-api` executable.
- Added config autodiscovery, Dockerfile generation, image build, and local
  Docker runtime commands.
- Kept the CLI embeddable so a parent CLI can mount it as a subcommand without
  this package claiming the parent command namespace.

### Milestone 5: Agent Protocol compatibility

- Added `GET /ok`, `GET /info`, and `GET /metrics`.
- Added search / count flows for assistants and threads.
- Added thread copy, prune, state, history, checkpoint lookup, and state
  mutation endpoints.
- Added protocol-v2 routes:
  `POST /threads/{thread_id}/commands`,
  `POST /threads/{thread_id}/stream`,
  `POST /threads/{thread_id}/stream/events`.
- Added run cancellation endpoints and stateless batch execution.
- Aligned `/info` feature flags with the live server surface.

</details>

## Prioritized Next Work

1. [ ] Add `/agents` route aliases over the existing assistant registry:
   `POST /agents/search`, `GET /agents/{agent_id}`, and
   `GET /agents/{agent_id}/schemas`.
2. [ ] Align request/response schemas with Agent Protocol field names:
   accept `agent_id` alongside `assistant_id`, expose `messages` where
   supported, and document any AgentSeek-specific response extensions.
3. [ ] Add restart-safe streaming replay and resumable event delivery:
   `Last-Event-ID`, durable event logs, and replay after process restarts.
4. [ ] Add first-class `X-Api-Key` auth for Agent Protocol clients.
5. [ ] Add assistant-config and auth integration examples:
   assistant `config`, JWT auth sample, and custom routes sample.
6. [ ] Add a minimal `agentseek.json` example and a fixture proving a plain
   `langgraph.json` project runs unchanged under `agentseek-api`.
7. [ ] Add full Store API parity.
8. [ ] Add crons / scheduler support.
9. [ ] Add MCP and A2A endpoint parity.

## Quickstart

Create a project config. `agentseek-api` auto-discovers `agentseek.json` first,
then `langgraph.json`.

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "chat": "chat.graph:graph"
  }
}
```

Run the API locally:

```bash
uv run agentseek-api dev
```

Run with an explicit config:

```bash
uv run agentseek-api dev --config ./langgraph.json
```

Check the server:

```bash
curl http://127.0.0.1:2024/health
curl http://127.0.0.1:2024/info
```

The low-level fallback is still available:

```bash
uv run uvicorn agentseek_api.main:app --reload --port 2024
```

## CLI

The canonical executable is `agentseek-api`:

```bash
agentseek-api <command> [arguments]
```

When running from this repository with `uv`, prefix commands with `uv run`.

### Shared Arguments

These options are accepted by runtime commands that load a graph config:

- `-c, --config PATH`: explicit `agentseek.json`, `langgraph.json`, or manifest path.
- `--env-file PATH`: dotenv-style file. Values override env values from the config file.

Without `--config`, the CLI searches the current working directory in this
order:

1. `agentseek.json`
2. `langgraph.json`
3. `AGENTSEEK_GRAPHS`, if it points to an existing file

### `dev`

Run the API with `uvicorn agentseek_api.main:app --reload`.

```bash
uv run agentseek-api dev [--config PATH] [--env-file PATH] [--host HOST] [--port PORT] [--no-reload]
```

Arguments:

- `--host HOST`: bind host. Default: `127.0.0.1`.
- `--port PORT`: bind port. Default: `2024`.
- `--no-reload`: disable uvicorn reload.
- `--n-jobs-per-worker`, `--debug-port`, `--wait-for-client`, `--no-browser`,
  `--studio-url`, `--allow-blocking`, `--tunnel`: parsed for LangGraph CLI
  compatibility, but currently rejected when set because this runtime does not
  implement those behaviors yet.

### `serve`

Run the API without reload. This is intended for containers and smoke tests.

```bash
uv run agentseek-api serve [--config PATH] [--env-file PATH] [--host HOST] [--port PORT]
```

Arguments:

- `--host HOST`: bind host. Default: `127.0.0.1`.
- `--port PORT`: bind port. Default: `2024`.

### `dockerfile`

Render a Dockerfile for the active config.

```bash
uv run agentseek-api dockerfile [--config PATH] [--env-file PATH] ./Dockerfile.agentseek
```

The generated Dockerfile copies the project to `/deps/agent`, installs the
package plus configured local dependencies, exports `AGENTSEEK_GRAPHS`, and
starts the API through the package entrypoint.

### `build`

Build a local Docker image from the generated runtime Dockerfile.

```bash
uv run agentseek-api build --config ./langgraph.json -t agentseek-api:dev
```

Arguments:

- `-t, --tag TAG`: required Docker image tag.
- `--platform PLATFORM`: Docker build platform.
- `--pull` / `--no-pull`: control base image pulling. Default: `--pull`.
- `--config`, `--env-file`: same config/env loading behavior as `dev`.

### `up`

Start a local Docker runtime for the active config.

```bash
uv run agentseek-api up --config ./langgraph.json --port 8123 --wait
```

Arguments:

- `--wait`: wait for `/health` after the container starts.
- `--base-image IMAGE`: base image to use when auto-building.
- `--image IMAGE`: use an existing image instead of auto-building.
- `--postgres-uri URI`: pass a PostgreSQL metadata database URI into the container.
- `-p, --port PORT`: host port. Default: `8123`.
- `--pull` / `--no-pull`: control build pulls. Default: `--pull`.
- `--recreate` / `--no-recreate`: replace an existing container with the same name.
- `--watch`, `--debugger-base-url`, `--debugger-port`, `--verbose`: parsed for
  LangGraph CLI compatibility, but currently rejected when set.
- `-d, --docker-compose`: accepted by the parser for compatibility; this CLI
  path runs Docker directly.

### `version`

Print both CLI and package names:

```bash
uv run agentseek-api version
```

`deploy` is intentionally not part of the current CLI surface.

## Config Files

`agentseek-api` accepts LangGraph-style graph mappings and AgentSeek manifest
entries. Graph references may point to:

- A module symbol: `package.module:graph`
- A relative Python file: `./graph.py:graph`
- A compiled graph object
- A zero-argument builder
- A `build_graph(checkpointer=...)` function
- A config-style factory that accepts a config dict

Useful config fields:

- `dependencies`: local package paths installed into generated Docker images.
- `graphs`: graph id to graph reference mapping.
- `env`: either a dotenv file path or an object of scalar environment values.
- `auth.path`: custom auth backend reference. This sets `AUTH_TYPE=custom` and
  `AUTH_MODULE_PATH`.
- `base_image`, `python_version`, `image_distro`, `pip_config_file`,
  `dockerfile_lines`: Docker build customization fields.

Endpoint-level LangGraph config keys such as `store`, `http`, and `api_version`
are tolerated by the CLI layer where possible, but they are not fully wired to
runtime behavior yet.

## Library And Embedding

Import the package when embedding the runtime in another Python process:

```python
from agentseek_api.main import create_app

app = create_app()
```

Parent CLI tools can also mount the command parser as a subcommand:

```python
import argparse
from agentseek_api.cli import register_subcommands, run_namespace

parser = argparse.ArgumentParser(prog="parent")
subparsers = parser.add_subparsers(dest="tool", required=True)
register_subcommands(subparsers, command_name="agentseek-api")

args = parser.parse_args()
raise SystemExit(run_namespace(args))
```

This lets a parent CLI expose commands such as:

```bash
parent agentseek-api dev --config ./langgraph.json
parent agentseek-api build --config ./langgraph.json -t my-api:dev
```

## Agent Protocol Compatibility

The runtime exposes the core
[Agent Protocol](https://github.com/langchain-ai/agent-protocol) surfaces for
agents, threads, runs, streaming, and thread protocol events. AgentSeek API's
current implementation still names the agent resource `Assistant`, so direct
`/agents` route aliases remain future work.

This comparison is based on
[Agent Protocol OpenAPI `0.1.6`](https://raw.githubusercontent.com/langchain-ai/agent-protocol/main/openapi.json).

Implemented surfaces include:

- System endpoints: `GET /health`, `GET /ok`, `GET /info`, `GET /metrics`.
- Assistants: create, list, search, count, get, patch, delete, graph metadata,
  schemas, subgraphs, versions, and latest.
- Threads: create, list, search, count, patch, delete, copy, prune, state,
  checkpoint lookup, history, stream, and command endpoints.
- Thread runs: create, list, get, wait, join, stream, resume, cancel, delete,
  and wait/stream creation shortcuts.
- Stateless runs: create, wait, stream, batch, and cancel.
- Protocol v2 event streaming through `POST /threads/{thread_id}/stream/events`.
- HITL interrupt/resume through `POST /threads/{thread_id}/runs/{run_id}/resume`.
- SSE streaming with `message_chunk` events for model-backed graph streams.

`GET /info` currently advertises:

```json
{
  "flags": {
    "assistants": true,
    "threads": true,
    "runs": true,
    "crons": false,
    "store": false,
    "a2a": false,
    "mcp": false,
    "protocol_v2": true
  }
}
```

<details>
<summary>Agent Protocol comparison</summary>

| Agent Protocol surface | Agent Protocol paths | AgentSeek API status |
| --- | --- | --- |
| System metadata | Not specified as core OpenAPI paths; implementations commonly expose health/info separately. | Implemented as `GET /health`, `GET /ok`, `GET /info`, and `GET /metrics`. |
| Agents | `POST /agents/search`, `GET /agents/{agent_id}`, `GET /agents/{agent_id}/schemas`. | Partially implemented under `assistants`: `POST /assistants/search`, `GET /assistants/{assistant_id}`, and `GET /assistants/{assistant_id}/schemas`. Direct `/agents` aliases are not implemented yet. |
| Threads | `POST /threads`, `POST /threads/search`, `GET /threads/{thread_id}`, `PATCH /threads/{thread_id}`, `DELETE /threads/{thread_id}`, `POST /threads/{thread_id}/copy`, `GET /threads/{thread_id}/history`. | Implemented. AgentSeek also exposes extensions such as `GET /threads`, `POST /threads/count`, `POST /threads/prune`, state/checkpoint routes, and protocol event routes. |
| Background runs | `POST /runs`, `POST /runs/search`, `GET /runs/{run_id}`, `DELETE /runs/{run_id}`, `POST /runs/{run_id}/cancel`, `GET /runs/{run_id}/wait`, `GET /runs/{run_id}/stream`. | Mostly implemented for stateless/background runs, including create, batch, wait, stream, and cancel. `POST /runs/search` is not implemented; current run listing is scoped under thread run routes. |
| Run creation shortcuts | `POST /runs/wait`, `POST /runs/stream`. | Implemented. |
| Thread run equivalents | Agent Protocol models stateful execution through `thread_id` on `RunCreate`, plus thread stream commands. | Implemented as explicit thread-run routes: `POST /threads/{thread_id}/runs`, `POST /threads/{thread_id}/runs/wait`, `POST /threads/{thread_id}/runs/stream`, wait/join/stream/get/delete/cancel, and resume. |
| Streaming protocol | `POST /threads/{thread_id}/commands`, `POST /threads/{thread_id}/stream`, `GET /threads/{thread_id}/stream`. | Implemented with `POST /threads/{thread_id}/commands`, `POST /threads/{thread_id}/stream`, `POST /threads/{thread_id}/stream/events`, and `GET /threads/{thread_id}/stream`. Current replay is in-memory, so restart-safe `Last-Event-ID` replay remains a gap. |
| Store | `PUT /store/items`, `GET /store/items`, `DELETE /store/items`, `POST /store/items/search`, `POST /store/namespaces`. | Not implemented. Current store object is internal only and should not be treated as Agent Protocol Store API parity. |
| Auth | Agent Protocol leaves deployment auth policy to the server implementation. | `AUTH_TYPE=noop` and `AUTH_TYPE=custom` are implemented. First-class `X-Api-Key` auth remains future work. |

Schema-level notes:

- Agent Protocol names the runnable resource `Agent` and uses `agent_id`; this
  runtime currently uses `Assistant` / `assistant_id`.
- Agent Protocol `RunCreate` accepts `agent_id`, optional `thread_id`, `input`,
  `messages`, `metadata`, `config`, `webhook`, `on_completion`,
  `on_disconnect`, and `if_not_exists`. AgentSeek currently accepts
  `assistant_id`, `input`, `metadata`, `config`, `context`, and
  `multitask_strategy`.
- Agent Protocol `ThreadCreate` accepts caller-provided `thread_id`, metadata,
  and `if_exists`. AgentSeek currently generates thread ids and also stores a
  `config` object.
- Agent Protocol `ThreadPatch` can patch `metadata`, `values`, and `messages`
  from an optional checkpoint. AgentSeek currently supports metadata patching
  and exposes separate state/checkpoint routes.
- Agent Protocol streaming events include ordered `eventId` / `seq` fields for
  reconnection and replay. AgentSeek emits protocol-v2 events with sequence
  data through the in-memory broker, but durable replay after restart is still
  not implemented.

</details>

### Current Gaps

These are known limitations, not hidden compatibility claims:

- Store API parity is not implemented. The runtime has internal checkpoint
  storage, but not the Agent Protocol Store API surface.
- Direct `/agents` route aliases are not implemented; the current resource
  surface is still named `/assistants`.
- `POST /runs/search` is not implemented.
- Full Agent Protocol request schema parity is not complete for `RunCreate`,
  `ThreadCreate`, and `ThreadPatch`.
- Cron/scheduled run API parity is not implemented.
- MCP and A2A endpoint parity are not implemented.
- First-class `X-Api-Key` hosted-server auth is not implemented. Current auth
  modes are `AUTH_TYPE=noop` and `AUTH_TYPE=custom`.
- Some LangGraph CLI flags are parsed for command-shape compatibility but
  rejected when used, as documented in the command sections above.
- Some endpoint-level `langgraph.json` keys are accepted by config parsing but
  deferred at runtime.

## Runtime Configuration

Metadata persistence uses SQLAlchemy:

- Prefer `METADATA_DB_URL`.
- `SEEKDB_URL` remains as a legacy fallback.
- `METADATA_DB_BACKEND=auto` infers the async driver:
  - PostgreSQL: `postgresql+asyncpg://...`
  - OceanBase / MySQL: `mysql+aiomysql://...`

Checkpoint persistence defaults to OceanBase / SeekDB via `OCEANBASE_*`
settings and `langchain-oceanbase`.

Auth modes:

- `AUTH_TYPE=noop`: default identity is `default_user`.
- `AUTH_TYPE=custom`: load a custom backend from
  `AUTH_MODULE_PATH=module:backend_symbol`.

Graph loading:

- `AGENTSEEK_GRAPHS=/abs/path/manifest.json` loads user-defined graphs at
  startup.
- Manifest entries override bundled graph ids.

## Live Provider Streaming

Real provider-backed streaming checks are intentionally manual so normal CI
does not spend model tokens.

- Workflow: `.github/workflows/live-provider-streaming.yml`
- Manifest: `examples/live_provider_graphs/manifest.json`
- Test target: `tests/integration/test_live_provider_streaming.py`

Repository settings:

- OpenAI-compatible:
  - variable `OPENAI_COMPAT_MODEL`
  - variable `OPENAI_COMPAT_BASE_URL`
  - secret `OPENAI_COMPAT_API_KEY`
- Anthropic-compatible:
  - variable `ANTHROPIC_COMPAT_MODEL`
  - variable `ANTHROPIC_COMPAT_BASE_URL`
  - secret `ANTHROPIC_COMPAT_API_KEY`

Run the workflow from GitHub Actions with `workflow_dispatch`. Dispatch inputs
may override model and base URL for one run. API keys stay in repository
secrets.

The proof target is at least two non-empty SSE `message_chunk` events from a
real provider-backed graph.

## How To Contribute

This repository uses a GitFlow-lite workflow:

- `main`: production-ready history only.
- `develop`: integration branch for ongoing development.
- `feature/<topic>`: branch from `develop`, PR back to `develop`.
- `release/<version>`: branch from `develop`, PR to `main`.
- `hotfix/<topic>`: branch from `main`, PR to `main`, then merge back into
  `develop`.

See `CONTRIBUTING.md` for the full branching and CI policy.

Useful local checks:

```bash
make test
make test-cov
make test-cli
make test-samples
```

Docker and live backend checks:

```bash
make test-cli-docker
make test-e2e
make test-seekdb
```

`test-seekdb` supports:

- `SEEKDB_MODE=auto`: prefer embedded SeekDB when `pylibseekdb` is available,
  otherwise use Docker.
- `SEEKDB_MODE=embed`: run with a custom embedded command.
- `SEEKDB_MODE=docker`: run against `seekdb`, `oceanbase`, or `mysql` Docker
  backends.

Example embedded SeekDB run:

```bash
SEEKDB_MODE=embed \
SEEKDB_EMBED_CMD="uv run python scripts/seekdb_embed_launcher.py" \
OCEANBASE_USER=root \
SEEKDB_URL="mysql+aiomysql://root:@127.0.0.1:2881/seekdb" \
make test-seekdb
```

CI runs the real backend validation in Docker across `seekdb`, `oceanbase`, and
`mysql`.
