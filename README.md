# AgentSeek API

Core Agent Protocol runtime for LangGraph / LangChain apps, with OceanBase as the default checkpoint backend.

## Quickstart

1. Copy env:
   - `cp .env.example .env`
2. Install:
   - `uv sync`
3. Run:
   - `uv run uvicorn agentseek_api.main:app --reload --port 2026`
4. Exercise the bundled sample graphs in-process:
   - `uv run python examples/run_sample_graphs.py`
   - Source for each sample lives under `examples/graphs/`; see `examples/README.md` for a tour.

## Local backend tests

- Unit + integration (mocked backend):
  - `make test`
- Unit + integration with coverage gate (90%):
  - `make test-cov`
- Real end-to-end API tests (live server + real backend):
  - `make test-e2e`
- Real SeekDB/OceanBase smoke test:
  - `make test-seekdb`

`test-seekdb` supports:
- `SEEKDB_MODE=auto` (default): tries embed command first if provided, otherwise Docker.
- `SEEKDB_MODE=embed` with `SEEKDB_EMBED_CMD="<your command>"`. A bundled launcher at `scripts/seekdb_embed_launcher.py` starts an in-process SeekDB via `pylibseekdb.open_with_service` (bootstrap user `root`, no password). Example invocation:
  ```
  SEEKDB_MODE=embed \
  SEEKDB_EMBED_CMD=".venv/bin/python scripts/seekdb_embed_launcher.py" \
  OCEANBASE_USER=root \
  SEEKDB_URL="mysql+aiomysql://root:@127.0.0.1:2881/seekdb" \
  make test-seekdb
  ```
- `SEEKDB_MODE=docker` with `SEEKDB_DOCKER_IMAGE` override.
- It now validates both:
  - direct checkpoint write/read smoke (`scripts/seekdb_checkpoint_smoke.py`)
  - real live API HTTP flow (`tests/e2e/e2e_live_http_flow.py`) against a started uvicorn server

## Notes

- Metadata persistence uses SQLAlchemy via `METADATA_DB_URL` (preferred) or `SEEKDB_URL` (legacy fallback).
- `METADATA_DB_BACKEND=auto` infers backend from URL scheme and forces async drivers:
  - PostgreSQL -> `postgresql+asyncpg://...`
  - OceanBase/MySQL -> `mysql+aiomysql://...`
- Checkpoint persistence defaults to OceanBase via `OCEANBASE_*` settings, using the same SeekDB deployment by default.
- Auth mode is explicit:
  - `AUTH_TYPE=noop` uses default identity `default_user`
  - `AUTH_TYPE=custom` requires `AUTH_MODULE_PATH=module:backend_symbol`
- Async checkpoint execution is intentionally blocked in milestone 1.
