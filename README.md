# AgentSeek API

Run LangGraph and LangChain apps behind a FastAPI runtime with a standalone
`agentseek-api` CLI.

> [!NOTE]
> AgentSeek API uses
> [Agent Protocol](https://github.com/langchain-ai/agent-protocol) as the main
> external compatibility reference. The current runtime already covers the core
> thread, run, cron, streaming, and protocol-v2 event flows. Some protocol surfaces
> are still pending, and agent resources are exposed through `/assistants`,
> direct `/agents` aliases, Streamable HTTP MCP, and LangSmith-style A2A
> endpoints. This is workable OSS parity for the core agent-server surfaces,
> not full LangSmith Agent Server parity.

Current release boundary:

- Implemented: assistants, threads, runs, crons, streaming, Store API, MCP,
  and A2A
- Explicitly not implemented: distributed runtime parity, assistant
  subgraph inspection, and assistant version-promotion workflows

## 🚀 Quickstart

### Prerequisites

- Python 3.12+
- `uv`

### Choose the right local loop

| Workflow | Use it when | Recommended command |
| --- | --- | --- |
| `langgraph dev` | You want the fastest mocked or in-memory local API loop for graph prototyping or Studio experimentation. | `langgraph dev` |
| `agentseek-api dev` | You want the real AgentSeek API surface with your actual MySQL-family / SeekDB / OceanBase-style persistence, auth, and Docker/runtime behavior. | `uv run agentseek-api dev` |

Use `langgraph dev` when you do not need real backend validation. Use
`agentseek-api dev` when you want to exercise the actual API contract this repo
ships.

### 1. Install dependencies

```bash
uv sync
```

### 2. Create a config file

`agentseek-api` looks for config in this order:

1. `AGENTSEEK_GRAPHS`, if it points to an existing file
2. `agentseek.json`
3. `langgraph.json`

Minimal `langgraph.json`:

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "chat": "chat.graph:graph"
  }
}
```

### 3. Start the local API

```bash
uv run agentseek-api dev
```

Run with an explicit config when needed:

```bash
uv run agentseek-api dev --config ./langgraph.json
```

When the server is ready it prints the local API, docs, and Studio URLs:

```text
> Ready!
>
> - API: http://localhost:2024
>
> - Docs: http://localhost:2024/docs
>
> - LangSmith Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### 4. Check that it is up

```bash
curl http://127.0.0.1:2024/health
curl http://127.0.0.1:2024/info
curl http://127.0.0.1:2024/openapi.json
```

## 🧰 CLI

The package installs `agentseek-api` as the standalone executable. It does not
install a top-level `agentseek` binary, which keeps the namespace free for a
parent CLI.

```bash
agentseek-api <command> [arguments]
```

When running from this repository, use `uv run agentseek-api ...`.

### Commands

| Command | What it does |
| --- | --- |
| `dev` | Start the API with reload for local development. |
| `serve` | Start the API without reload for containers or smoke tests. |
| `worker` | Start the Redis-backed run worker. |
| `dockerfile` | Generate a runtime Dockerfile for the active config. |
| `build` | Build a Docker image for the active config. |
| `up` | Start a local Docker runtime for the active config. |
| `version` | Print the installed package version. |

### Shared arguments

- `-c, --config PATH`: explicit `agentseek.json`, `langgraph.json`, or manifest
  path
- `--env-file PATH`: dotenv-style file loaded into the runtime environment

### Common usage

```bash
uv run agentseek-api dev
uv run agentseek-api serve --config ./langgraph.json --port 8080
uv run agentseek-api worker --config ./langgraph.json
uv run agentseek-api dockerfile --config ./langgraph.json ./Dockerfile.agentseek
uv run agentseek-api build --config ./langgraph.json -t agentseek-api:dev
uv run agentseek-api up --config ./langgraph.json --port 8123 --wait
uv run agentseek-api version
```

### Command notes

- `dev`
  - Default host: `127.0.0.1`
  - Default port: `2024`
  - Use `--no-reload` to disable reload
  - Use `--no-browser` to suppress automatic Studio launch
  - Use `--studio-url` to point at a different LangSmith / Studio origin
- `serve`
  - Same host and port options as `dev`
- `worker`
  - Requires `EXECUTOR_BACKEND=redis`
  - Uses `REDIS_URL` plus the queue keys below
  - Redis durable execution currently uses a single active worker lease at a time
  - Run and thread stream replay continues from persisted state after worker restarts
- `scheduler`
  - Triggers persisted cron jobs that are due for execution
  - Run alongside the API server and worker when cron support is enabled
- `build`
  - Use `-t, --tag` to set the image tag
  - Supports `--platform`, `--pull`, and `--no-pull`
- `up`
  - Supports `--wait`, `--image`, `--base-image`, `--postgres-uri`,
    `--recreate`, and `--no-recreate`

Some LangGraph CLI-shaped flags are parsed for command compatibility but
rejected when their runtime behavior is not implemented yet. For mocked,
in-memory, or tunneled local workflows, prefer `langgraph dev`.

## ✨ Features

- ⚙️ Standalone CLI plus embeddable subcommand registration for parent CLIs
- 🔌 Manifest-driven graph loading through `agentseek.json`, `langgraph.json`,
  or `AGENTSEEK_GRAPHS`
- 🌊 SSE streaming with `message_chunk` events
- 🧰 MCP tool exposure for registered graphs over Streamable HTTP
- 🤝 A2A assistant endpoints with agent-card discovery, streaming, and task lookup/cancel
- 🧵 Thread, run, wait, cancel, history, state, and protocol-v2 stream flows
- ⏰ Persisted cron APIs plus scheduler dispatch for stateless and thread-bound runs
- 🤖 Agent resources exposed through both `/assistants` and `/agents`
- 🧑‍💻 Human-in-the-loop resume through
  `POST /threads/{thread_id}/runs/{run_id}/resume`
- 🗄️ SeekDB / OceanBase-first checkpoint persistence via
  `langchain-oceanbase`
- 📦 Redis-backed durable execution with a dedicated worker process
- ♻️ Persisted run and thread stream replay for resume-after-restart flows
- 🔐 `noop` and custom auth backends
- 🐳 Dockerfile generation, image build, and local Docker runtime helpers
- 🧪 Real backend CI coverage across MySQL, SeekDB, OceanBase, and Redis runtime paths
- 🧪 Manual provider-backed streaming checks for live SSE proof

## 🎯 Compatibility Scope

Treat AgentSeek API as a practical OSS-compatible core for Agent Server-style
apps.

- Shipped: assistant CRUD, thread/run lifecycle APIs, resumable SSE streams,
  cron APIs and scheduler dispatch, Store API, MCP, A2A, Redis-backed durable
  execution, and Docker/runtime helpers
- Intentionally missing: distributed runtime orchestration parity, full
  assistant version management, assistant subgraph inspection, and full
  assistant helper parity beyond the core CRUD and schema flows

## 🚚 Deployment Roles

Cron-enabled deployments run three long-lived roles:

- API: serves `/assistants`, `/threads`, `/runs`, `/runs/crons`, `/info`, and the other HTTP surfaces
- Worker: executes Redis-backed queued runs and resumes persisted stream state after restarts
- Scheduler: claims due cron jobs and submits their runs into the runtime

For local development against a real backend, use `uv run agentseek-api dev`.
For mocked or in-memory graph iteration, use `langgraph dev` instead. For
durable cron execution, run the API server, worker, and scheduler together
against the same database and Redis instance.

## 🗂️ Config

Graph references may point to:

- a module symbol such as `package.module:graph`
- a relative Python file such as `./graph.py:graph`
- a compiled graph object
- a zero-argument builder
- a `build_graph(checkpointer=...)` function
- a config-style factory that accepts a config dict

Useful config fields:

- `dependencies`: local package paths installed into generated Docker images
- `graphs`: graph id to graph reference mapping
- `env`: either a dotenv file path or an object of scalar environment values
- `auth.path`: custom auth backend reference
- `auth.openapi`: OpenAPI `securitySchemes` and `security` metadata for auth
- `auth.disable_studio_auth`: disables the Studio auth bypass described below
- `http.disable_mcp`: disable the MCP endpoint
- `http.disable_a2a`: disable the A2A endpoint and agent-card discovery route
- `base_image`, `python_version`, `image_distro`, `pip_config_file`,
  `dockerfile_lines`: Docker build customization fields

Endpoint-level LangGraph config keys such as `http` and `api_version` are
tolerated by the CLI layer where possible. Store config is used by the HTTP
Store API and the injected LangGraph `BaseStore` runtime for TTL and semantic
search. This repo uses the published `langchain-oceanbase==0.5.0` package from
PyPI.

Config-driven custom auth can live in `agentseek.json` or `langgraph.json`:

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "auth": {
    "path": "./auth.py:auth",
    "openapi": {
      "securitySchemes": {
        "apiKeyAuth": {
          "type": "apiKey",
          "in": "header",
          "name": "X-API-Key"
        }
      },
      "security": [{ "apiKeyAuth": [] }]
    },
    "disable_studio_auth": false
  }
}
```

### Studio and docs behavior

- FastAPI docs stay available at `/docs`, `/redoc`, and `/openapi.json`
- Studio connects through the same local API base URL printed by
  `agentseek-api dev`
- When auth is configured, `agentseek-api dev` accepts loopback Studio requests
  carrying `x-auth-scheme: langsmith`
- Set `auth.disable_studio_auth` to `true` if Studio should use the same normal
  API auth path as every other client during `dev`
- If you only need a mocked local API server for Studio experiments, use
  `langgraph dev` instead of AgentSeek

## 🔌 MCP

AgentSeek API exposes registered graphs as MCP tools through a stateless
Streamable HTTP endpoint at `/mcp`.

### Behavior

- Transport: Streamable HTTP
- Session model: stateless
- Auth: same as the rest of the API
- Paths: `/mcp` and `/mcp/` are both accepted
- Discovery source: registered graphs from `agentseek.json`,
  `langgraph.json`, or `AGENTSEEK_GRAPHS`
- Enablement: MCP is enabled by default; `http.disable_mcp: true` disables it
- Safety: if the active config file exists but cannot be parsed, MCP stays disabled until the config is fixed

Graph object entries can carry MCP-facing metadata directly in the manifest:

```json
{
  "graphs": {
    "docs_agent": {
      "graph": "./docs_agent.py:graph",
      "name": "docs_agent",
      "description": "Answers documentation questions",
      "input_schema": {
        "type": "object",
        "properties": {
          "question": { "type": "string" }
        },
        "required": ["question"]
      },
      "output_schema": {
        "type": "object",
        "properties": {
          "answer": { "type": "string" }
        },
        "required": ["answer"]
      }
    }
  }
}
```

When these fields are omitted, AgentSeek falls back to:

- tool name = graph id
- description = empty string
- input schema = `{"type": "object"}`
- output schema = `{"type": "object"}`

### Disable MCP

Set `http.disable_mcp` in your config file:

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "http": {
    "disable_mcp": true
  }
}
```

### Python client example

```python
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async with httpx.AsyncClient(
    headers={"X-API-Key": "secret"},
    trust_env=False,
) as http_client:
    async with streamable_http_client(
        url="http://127.0.0.1:2024/mcp",
        http_client=http_client,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(tools)
```

This milestone only covers exposing AgentSeek graphs as MCP tools. It does not
add outbound MCP client support inside AgentSeek graphs.

## 🤝 A2A

AgentSeek API exposes assistants through a LangSmith-style A2A endpoint at
`/a2a/{assistant_id}` plus agent-card discovery at
`/.well-known/agent-card.json?assistant_id={assistant_id}`.

### Behavior

- Methods: `message/send`, `message/stream`, `tasks/get`, `tasks/cancel`
- Agent card discovery: assistant-scoped `/.well-known/agent-card.json`
- Auth: same as the rest of the API
- Paths: `/a2a/{assistant_id}` only
- Discovery source: assistants backed by message-compatible graphs
- Threading: incoming `contextId` is forwarded as LangGraph `thread_id`
- Enablement: A2A is enabled by default; `http.disable_a2a: true` disables both the RPC endpoint and agent-card discovery
- Safety: if the active config file exists but cannot be parsed, A2A stays disabled until the config is fixed

### Disable A2A

Set `http.disable_a2a` in your config file:

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "http": {
    "disable_a2a": true
  }
}
```

### Python client example

```python
import httpx

assistant_id = "<assistant-id>"
payload = {
    "jsonrpc": "2.0",
    "id": "send-1",
    "method": "message/send",
    "params": {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "hello"}],
            "messageId": "msg-1",
        }
    },
}

with httpx.Client(headers={"X-API-Key": "secret"}, trust_env=False) as client:
    response = client.post(
        f"http://127.0.0.1:2024/a2a/{assistant_id}",
        json=payload,
        headers={"Accept": "application/json"},
    )
    response.raise_for_status()
    print(response.json())
```

This endpoint targets practical LangSmith parity for assistant-to-assistant
messaging and SDK interoperability. Task tracking is in-process, so task lookup
and cancellation only apply to tasks created on the current API process.

## 📚 Use As A Library

Embed the FastAPI app directly:

```python
from agentseek_api.main import create_app

app = create_app()
```

Mount the CLI under a parent tool without giving up the standalone
`agentseek-api` binary:

```python
import argparse
from agentseek_api.cli import register_subcommands, run_namespace

parser = argparse.ArgumentParser(prog="parent")
subparsers = parser.add_subparsers(dest="tool", required=True)
register_subcommands(subparsers, command_name="api")

args = parser.parse_args()
raise SystemExit(run_namespace(args))
```

That lets a parent CLI expose commands like:

```bash
parent api dev --config ./langgraph.json
parent api build --config ./langgraph.json -t my-api:dev
```

## 🏗️ Runtime Notes

- Metadata persistence uses `METADATA_DB_URL` when it is set
- Otherwise the metadata database URL is resolved from `SEEKDB_URL` or the
  `OCEANBASE_*` connection settings
- Run execution defaults to `EXECUTOR_BACKEND=inline`
- Set `EXECUTOR_BACKEND=redis` and start `agentseek-api worker` to hand off
  runs through Redis
- Redis queue settings:
  - `REDIS_URL=redis://127.0.0.1:6379/0`
  - `REDIS_RUN_QUEUE_KEY=agentseek:runs:pending`
  - `REDIS_RUN_PROCESSING_KEY=agentseek:runs:processing`
  - `REDIS_WORKER_LOCK_KEY=agentseek:worker:active`
  - `REDIS_WORKER_LOCK_TTL_SECONDS=30`
- `METADATA_DB_BACKEND=auto` normalizes drivers:
  - PostgreSQL: `postgresql+asyncpg://...`
  - OceanBase / MySQL: `mysql+aiomysql://...`
- Checkpoint persistence defaults to OceanBase / SeekDB settings
- Auth modes:
  - `AUTH_TYPE=noop`
  - `AUTH_TYPE=custom` with `AUTH_MODULE_PATH=module:backend_symbol`
  - `AUTH_TYPE=api_key` with `AUTH_API_KEYS=key=user_id[,key2=user2]`
  - `AUTH_TYPE=jwt` with `AUTH_JWT_SECRET`, optional
    `AUTH_JWT_ALGORITHM=HS256`, and `sub` as the user identity
- Assistant management, thread, and run endpoints enforce configured auth.

### Durable execution

- Redis mode persists run stream events and protocol stream events into the
  metadata database so stream replay does not depend on API-process memory.
- Interrupted runs can be resumed after worker restart as long as Redis and the
  metadata database stay available.
- The worker owns a renewable Redis lease and exits if that lease is lost,
  which prevents split-brain execution.

## 🧭 Examples

- `examples/minimal_agentseek/agentseek.json`: minimal first-time config
- `examples/assistant_config/`: assistant config/context/metadata starter
- `examples/auth/custom_backend.py`: custom auth backend
- `examples/auth/jwt.md`: JWT auth environment contract
- `examples/custom_routes/app.py`: mounting custom FastAPI routes around the
  AgentSeek API app

## 🧪 Contributing

This repository uses a GitFlow-lite workflow:

- `main`: production-ready history only
- `develop`: integration branch for ongoing development
- `feature/<topic>`: branch from `develop`, PR back to `develop`
- `release/<version>`: branch from `develop`, PR to `main`
- `hotfix/<topic>`: branch from `main`, PR to `main`, then merge back into
  `develop`

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
make test-redis-docker
```

GitHub Actions also runs the Docker-backed backend matrix against MySQL,
SeekDB, and OceanBase, including the dedicated `Redis Durable Execution`
workflow jobs.

For local embedded SeekDB smoke coverage, install the optional extra first:

```bash
uv sync --dev --extra embedded
```

This repository intentionally uses two GitHub Actions workflows for CI:

- `.github/workflows/ci.yml`
  - Always-on repository CI for pull requests and normal branch pushes
  - Fast enough to run by default, with no external model-provider spend
  - Covers unit and integration tests, CLI compatibility, sample graphs, Docker
    runtime checks, MySQL-family checkpoint validation, PostgreSQL metadata
    validation, and Redis durable execution
  - Uses local or Docker-backed dependencies that GitHub can provision inside
    the job

- `.github/workflows/live-provider-streaming.yml`
  - Dedicated real-model workflow for provider-backed proof
  - Runs only on manual dispatch or the nightly schedule
  - Uses the repo variables and secrets for OpenAI-compatible and
    Anthropic-compatible providers
  - Exists separately so default PR CI stays fast, deterministic, and free from
    external provider cost/rate-limit flake

The design intent is:

- `ci.yml` proves the product logic, storage/runtime integrations, and backend
  compatibility without depending on a live model provider
- `live-provider-streaming.yml` proves that the same API surfaces still work
  when a real provider is in the loop

The live-provider workflow is the canonical proof for real SSE
`message_chunk` events from provider-backed graphs, and it now also covers
provider-backed Store, MCP, and HITL flows in a tiered backend matrix:

- SeekDB: full Streaming + Store + MCP + HITL acceptance
- OceanBase: full Streaming + Store + MCP + HITL acceptance
- MySQL: Streaming + HITL compatibility
- PostgreSQL metadata: Streaming + MCP compatibility while runtime
  checkpointer/store still use a MySQL-family backend

Workflow behavior:

- Manual dispatch can target one provider, one backend tier, or the full matrix
- The nightly schedule runs the full provider/backend matrix
- Provider configuration is validated before the suite runs
- Backend capabilities are gated by tier, so MySQL does not run Store/MCP and
  PostgreSQL metadata does not pretend to replace the runtime MySQL-family
  store/checkpointer path
- Logs are uploaded for every lane, including failures

Use `ci.yml` for the normal development signal, and use
`live-provider-streaming.yml` when you need explicit proof that real providers
still satisfy the intended streaming, Store, MCP, and HITL contracts.

## 🗺️ Future Work

1. [x] Add Redis-backed task queue and worker handoff for durable run execution
2. [ ] Deepen Agent Protocol schema parity beyond the current alias coverage
3. [x] Add crons and scheduler support
4. [ ] Add distributed runtime parity beyond the current single-worker Redis flow
5. [ ] Add assistant version history and subgraph inspection support
6. [x] Add MCP and A2A endpoint parity
