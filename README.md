# AgentSeek API

Run LangGraph and LangChain apps behind a FastAPI runtime with a standalone
`agentseek-api` CLI.

> [!NOTE]
> AgentSeek API uses
> [Agent Protocol](https://github.com/langchain-ai/agent-protocol) as the main
> external compatibility reference. The current runtime already covers the core
> thread, run, streaming, and protocol-v2 event flows. Some protocol surfaces
> are still pending, and agent resources are currently exposed as
> `assistants` rather than direct `/agents` aliases.

## 🚀 Quickstart

### Prerequisites

- Python 3.12+
- `uv`

### 1. Install dependencies

```bash
uv sync
```

### 2. Create a config file

`agentseek-api` looks for config in this order:

1. `agentseek.json`
2. `langgraph.json`
3. `AGENTSEEK_GRAPHS`, if it points to an existing file

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

### 4. Check that it is up

```bash
curl http://127.0.0.1:2024/health
curl http://127.0.0.1:2024/info
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
- `serve`
  - Same host and port options as `dev`
- `build`
  - Use `-t, --tag` to set the image tag
  - Supports `--platform`, `--pull`, and `--no-pull`
- `up`
  - Supports `--wait`, `--image`, `--base-image`, `--postgres-uri`,
    `--recreate`, and `--no-recreate`

Some LangGraph CLI-shaped flags are parsed for command compatibility but
rejected when their runtime behavior is not implemented yet.

## ✨ Features

- ⚙️ Standalone CLI plus embeddable subcommand registration for parent CLIs
- 🔌 Manifest-driven graph loading through `agentseek.json`, `langgraph.json`,
  or `AGENTSEEK_GRAPHS`
- 🌊 SSE streaming with `message_chunk` events
- 🧵 Thread, run, wait, cancel, history, state, and protocol-v2 stream flows
- 🧑‍💻 Human-in-the-loop resume through
  `POST /threads/{thread_id}/runs/{run_id}/resume`
- 🗄️ SeekDB / OceanBase-first checkpoint persistence via
  `langchain-oceanbase`
- 🔐 `noop` and custom auth backends
- 🐳 Dockerfile generation, image build, and local Docker runtime helpers
- 🧪 Real backend validation and manual provider-backed streaming checks

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
- `auth.disable_studio_auth`: accepted for LangGraph config compatibility
- `base_image`, `python_version`, `image_distro`, `pip_config_file`,
  `dockerfile_lines`: Docker build customization fields

Endpoint-level LangGraph config keys such as `http` and `api_version` are
tolerated by the CLI layer where possible. Store config is used by the HTTP
Store API for TTL and custom embedding-function setup; graph-injected
`BaseStore` runtime support remains future work because
`langchain-oceanbase==0.4.0` does not expose a store adapter yet.

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
```

Real provider-backed streaming proof stays in the manual workflow
`.github/workflows/live-provider-streaming.yml`. That workflow is the canonical
check for real SSE `message_chunk` events from provider-backed graphs.

## 🗺️ Future Work

1. [ ] Add Redis-backed task queue and worker handoff for durable run execution
2. [ ] Add graph-injected `BaseStore` runtime support once `langchain-oceanbase`
   exposes a durable store adapter; the current HTTP Store API persists through
   AgentSeek metadata tables
3. [ ] Add provider-managed semantic embedding strings for Store API indexing
4. [ ] Add direct `/agents` aliases and deeper Agent Protocol schema parity
5. [ ] Add crons and scheduler support
6. [ ] Add MCP and A2A endpoint parity
