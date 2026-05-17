# AgentSeek API

Core Agent Protocol runtime for LangGraph / LangChain apps, with OceanBase as the default checkpoint backend.

## Quickstart

1. Copy env:
   - `cp .env.example .env`
2. Install:
   - `uv sync`
3. Run:
   - `uv run agentseek dev`
   - optional explicit config: `uv run agentseek dev --config ./langgraph.json`
   - raw fallback: `uv run uvicorn agentseek_api.main:app --reload --port 2024`
   - generate a container file from an explicit config: `uv run agentseek dockerfile --config ./examples/external_graph/manifest.json ./Dockerfile.agentseek`
   - plan a local image build: `uv run agentseek build --config ./examples/external_graph/manifest.json -t agentseek-api:dev`
4. Exercise the bundled sample graphs in-process:
   - `uv run python examples/run_sample_graphs.py`
   - `uv run python examples/external_graph/run.py`
   - Source for each sample lives under `examples/graphs/`; see `examples/README.md` for a tour.
5. Optional: register external graphs with a manifest:
   - `export AGENTSEEK_GRAPHS=$PWD/examples/external_graph/manifest.json`
   - restart the server, then create assistants with `graph_id: "external_hello"`
   - the loader accepts both agentseek-style graph objects and basic `langgraph.json` graph mappings such as `"graphs": {"chat": "./chat.py:graph"}`

## Manual Live Provider Streaming Check

This repo now includes a dispatch-only GitHub Actions workflow for proving
real token streaming against provider endpoints without putting model spend
on every PR:

- workflow: `.github/workflows/live-provider-streaming.yml`
- manifest: `examples/live_provider_graphs/manifest.json`
- test target: `tests/integration/test_live_provider_streaming.py`

Set these GitHub repository settings before running the workflow:

- OpenAI-compatible endpoint:
  - repository variable `OPENAI_COMPAT_MODEL`
  - repository variable `OPENAI_COMPAT_BASE_URL`
  - repository secret `OPENAI_COMPAT_API_KEY`
- Anthropic-compatible endpoint:
  - repository variable `ANTHROPIC_COMPAT_MODEL`
  - repository variable `ANTHROPIC_COMPAT_BASE_URL`
  - repository secret `ANTHROPIC_COMPAT_API_KEY`

Then run the workflow from the Actions tab with `workflow_dispatch`.
You can override the model or base URL for one run by filling the optional
dispatch inputs; API keys remain in secrets only.

Notes:

- For OpenAI-compatible providers, `OPENAI_COMPAT_BASE_URL` usually includes
  the provider's `/v1` prefix.
- For Anthropic-compatible providers, set `ANTHROPIC_COMPAT_BASE_URL` to the
  provider's Anthropic-style API root.
- The live check passes only when `/stream` replays at least two non-empty
  `message_chunk` SSE events from the real model-backed graph.

## Local backend tests

- Unit + integration (mocked backend):
  - `make test`
- Unit + integration with coverage gate (90%):
  - `make test-cov`
- In-process sample graph + API smoke tests:
  - `make test-samples`
- CLI Docker smoke for `dockerfile` + `build` + `up`:
  - `make test-cli-docker`
- Real end-to-end API tests (live server + real backend):
  - `make test-e2e`
- Real SeekDB/OceanBase smoke test:
  - `make test-seekdb`

`test-seekdb` supports:
- `SEEKDB_MODE=auto` (default): prefers embedded SeekDB when `pylibseekdb` is available, otherwise falls back to Docker.
- `SEEKDB_MODE=embed` with `SEEKDB_EMBED_CMD="<your command>"`. A bundled launcher at `scripts/seekdb_embed_launcher.py` starts an in-process SeekDB via `pylibseekdb.open_with_service` (bootstrap user `root`, no password). Example invocation:
  ```
  SEEKDB_MODE=embed \
  SEEKDB_EMBED_CMD="uv run python scripts/seekdb_embed_launcher.py" \
  OCEANBASE_USER=root \
  SEEKDB_URL="mysql+aiomysql://root:@127.0.0.1:2881/seekdb" \
  make test-seekdb
  ```
- `SEEKDB_MODE=docker` with `SEEKDB_DOCKER_BACKEND=seekdb|oceanbase|mysql` and optional `SEEKDB_DOCKER_IMAGE` override.
- It now validates both:
  - direct checkpoint write/read smoke (`scripts/seekdb_checkpoint_smoke.py`)
  - real live API HTTP flows (`tests/e2e/e2e_live_http_flow.py`, `tests/e2e/e2e_live_http_multi_graph.py`, and `tests/e2e/e2e_live_http_resume_flow.py`) against a started uvicorn server
  - the pytest-marked e2e suite (`make test-e2e`) against a real SeekDB/OceanBase backend

CI note:
- Local quick verification should prefer embedded SeekDB.
- GitHub Actions runs the real backend validation in Docker against a matrix of `seekdb`, `oceanbase`, and `mysql`.

## Branching Model

This repository uses a GitFlow-lite workflow:

- `main`: production-ready history only
- `develop`: integration branch for ongoing development
- `feature/<topic>`: branch from `develop`, PR back to `develop`
- `release/<version>`: branch from `develop`, PR to `main`
- `hotfix/<topic>`: branch from `main`, PR to `main`, then merge back into `develop`

See [CONTRIBUTING.md](/Users/zhl/workspaces/agentseek-api/CONTRIBUTING.md) for the full branching and CI policy.

## Notes

- Metadata persistence uses SQLAlchemy via `METADATA_DB_URL` (preferred) or `SEEKDB_URL` (legacy fallback).
- `METADATA_DB_BACKEND=auto` infers backend from URL scheme and forces async drivers:
  - PostgreSQL -> `postgresql+asyncpg://...`
  - OceanBase/MySQL -> `mysql+aiomysql://...`
- Checkpoint persistence defaults to OceanBase via `OCEANBASE_*` settings, using the same SeekDB deployment by default.
- `AGENTSEEK_GRAPHS=/abs/path/manifest.json` loads user-defined graphs at startup. Manifest entries override bundled IDs.
- `agentseek dev` / `agentseek serve` auto-discover `agentseek.json` first, then `langgraph.json`, and export the selected file through `AGENTSEEK_GRAPHS`.
- `agentseek dockerfile` renders a container entrypoint that keeps the selected config mounted at `/deps/agent/...` and starts the API with `agentseek serve`.
- CI now smoke-tests the Docker CLI path by rendering a Dockerfile, building the image with `agentseek build`, starting it with `agentseek up`, and probing `/health` and `/info`.
- Graph definitions are compatible with LangGraph-style entries: relative `./file.py:symbol`, module-path `package.module:symbol`, compiled graph variables, zero-arg builders, `build_graph(checkpointer=...)`, and config-style factories that rebuild from a config dict.
- Auth mode is explicit:
  - `AUTH_TYPE=noop` uses default identity `default_user`
  - `AUTH_TYPE=custom` requires `AUTH_MODULE_PATH=module:backend_symbol`
- Runs can surface `status=interrupted` plus an `interrupts` payload; resume them with `POST /threads/{thread_id}/runs/{run_id}/resume`.
