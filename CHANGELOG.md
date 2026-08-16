# Changelog

Notable changes to AgentSeek API are documented in this file.

## Unreleased

## 0.2.3 - 2026-08-17

### Fixed

- When `METADATA_DB_BACKEND=sqlite`, store completed-run snapshots in the
  configured SQLite metadata database instead of constructing an OceanBase
  saver and attempting a MySQL-family connection.

### Upgrade notes

- No manual schema or configuration migration is required. The
  `agentseek_checkpoints` table is created idempotently at startup.
- This fix covers completed-run snapshots only. LangGraph checkpoint state
  remains in memory when the metadata backend is SQLite.

## 0.2.2 - 2026-08-16

### Highlights

- Added bounded parallel Redis job execution within a worker process. Set
  `WORKER_CONCURRENT_JOBS` to control the limit; the default is `10`.
- Aligned assistant and run `config.configurable` / `context` handling with the
  LangGraph API contract, including assistant defaults and legacy configurable
  access for graphs without a context schema.
- Defined one deterministic environment-ownership contract for the `dev`,
  `serve`, `worker`, and `scheduler` host runtimes.

### Fixed

- Made Redis job acknowledgement conditional on ownership of the active worker
  lease, and stopped scheduling safely when a job fails or the lease is lost.
- Rejected client requests that provide both non-empty `config.configurable` and
  `context`, mirrored either input into the other representation, and merged
  assistant configuration into run configuration without discarding nested
  defaults.
- Made inherited host environment keys, including explicit empty strings,
  authoritative over config and CLI dotenv sources.
- Parsed each dotenv source independently with strict malformed-file handling,
  and distinguished valueless `KEY` from explicit empty `KEY=`.
- Started `worker` and `scheduler` in fresh child processes so their settings
  are built after the resolved environment is installed.
- Added bounded signal forwarding and descendant-process cleanup for `serve`,
  `worker`, and `scheduler` across supported platforms.
- Kept version, help, and Dockerfile rendering independent of runtime settings
  validation.
- Fell back to an ASCII CLI banner when a legacy Windows console cannot encode
  the Unicode banner.

### Upgrade notes

- Redis workers now execute up to `10` jobs concurrently by default. Set
  `WORKER_CONCURRENT_JOBS=1` to retain the previous serial behavior.
- Clients must not send both non-empty `config.configurable` and `context` in
  one assistant or run request; such requests now return HTTP 400. Prefer
  `context` for new integrations.
- For `dev`, `serve`, `worker`, and `scheduler`, dotenv interpolation is
  supported only against the inherited environment and earlier bindings in the
  same physical file.
- For those host runtime commands, missing, malformed, and non-UTF-8 dotenv
  sources fail closed with status 2 before a runtime child starts.
- Dependency resolution now requires SQLAlchemy 2.0.12 or newer, LangGraph
  1.0.6 or newer, LangChain Core 1.2.5 or newer, and MCP 1.27.1 through the 1.x
  series. `python-dotenv` 1.0 through 1.2 is now a direct dependency.

## 0.2.1 - 2026-07-14

### Fixed

- Preserved empty JSON arrays in Redis thread-stream events by avoiding a Lua
  cjson decode/encode round trip, while keeping generated envelope fields
  authoritative.
- Mirrored completed tool, human, and system messages to `messages-tuple` when
  requested so LangGraph SDK clients receive tool results before final-answer
  streaming completes.

### Upgrade notes

- No schema or configuration migration is required from 0.2.0.

## 0.2.0 - 2026-07-12

### Highlights

- Expanded the Crons API to align its request, response, filtering, sorting,
  selection, and lifecycle behavior with the LangGraph Platform contract.
- Moved Redis-worker run and protocol event persistence from SQL to bounded
  Redis Streams, removing metadata-database writes from the streaming hot path.
- Preserved UTF-8 output across SSE, wait, and A2A JSON responses while keeping
  active thread event retention bounded without dropping live events.
- Upgraded checkpoint integration to `langchain-oceanbase` 0.6.0.

### Upgrade notes

- Drain active Redis-worker runs before upgrading. Stream-event rows written to
  SQL by earlier versions are not imported into Redis Streams.
- Redis stream replay is bounded by `REDIS_STREAM_MAXLEN` and
  `REDIS_STREAM_TTL_SECONDS`; review these settings for long-running workloads.
