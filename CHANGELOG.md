# Changelog

Notable changes to AgentSeek API are documented in this file.

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
