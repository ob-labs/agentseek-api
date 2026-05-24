# A2A Endpoint Design

## Goal

Add LangSmith-style A2A endpoint support to `agentseek-api` so assistants can be addressed through:

- `POST /a2a/{assistant_id}`
- `GET /.well-known/agent-card.json?assistant_id={assistant_id}`

The implementation should match the documented LangSmith behavior closely enough for practical parity, while staying strict about graph compatibility and configuration validity.

## Scope

This tranche includes:

- A2A endpoint enablement behind config
- Assistant-scoped A2A routing
- Agent Card discovery
- JSON-RPC handling for:
  - `message/send`
  - `message/stream`
  - `tasks/get`
  - `tasks/cancel`
- Assistant-first metadata in the Agent Card
- Integration and e2e verification
- README updates

This tranche does not include:

- Stronger durability guarantees than current LangSmith parity requires
- Cross-process or restart-safe A2A task recovery
- Broader protocol surface beyond the methods listed above
- Implicit coercion for non-message-shaped graphs

## Product Decisions

### Assistant-first identity

The A2A contract is assistant-scoped, not graph-scoped. The Agent Card should therefore present:

- `name` from the assistant row
- `description` from the assistant row
- endpoint URL derived from `/a2a/{assistant_id}`

Graph-derived metadata should still drive capability details such as accepted input structure and output behavior.

### LangSmith-style parity

`tasks/get` and `tasks/cancel` should match the practical behavior of LangSmith’s A2A surface, but do not need stronger guarantees. In-process task tracking is acceptable for this milestone.

### Strict compatibility

Only assistants backed by A2A-compatible graphs should be callable through `/a2a/{assistant_id}`. If a graph cannot accept message-shaped input, the server should reject the request clearly instead of attempting lossy adaptation.

## External Contract

### Routes

- `POST /a2a/{assistant_id}`
  - accepts A2A JSON-RPC requests
  - supports `message/send`, `message/stream`, `tasks/get`, `tasks/cancel`
- `GET /.well-known/agent-card.json?assistant_id={assistant_id}`
  - returns the assistant’s Agent Card

### Feature flags

Expose A2A enablement through:

- `/info`
- `/metrics?format=json`

The `flags.a2a` field should reflect startup-time route enablement.

### Configuration

Use `http.disable_a2a` in the active config file, matching the MCP pattern.

Rules:

- missing config: A2A enabled by default
- missing `http` section: A2A enabled by default
- `http.disable_a2a: true`: A2A disabled
- malformed config file: A2A disabled
- malformed `http` section: A2A disabled
- non-boolean `disable_a2a`: A2A disabled

This is fail-closed behavior.

## Architecture

### Main module split

Add a dedicated A2A server module, parallel to `mcp_server.py`, instead of folding A2A into the existing REST routers. This keeps protocol adaptation isolated and avoids leaking A2A-specific request and response shaping into the primary REST handlers.

Expected pieces:

- `src/agentseek_api/core/a2a_config.py`
  - startup-time config gating
- `src/agentseek_api/a2a_server.py`
  - request parsing
  - task registry
  - assistant resolution
  - graph compatibility checks
  - JSON-RPC response shaping
  - streaming response shaping
- `src/agentseek_api/main.py`
  - startup registration for A2A routes
  - lifespan wiring if task cleanup hooks are needed
  - feature flag reporting

### Reuse points

The A2A server should reuse existing internal seams where possible:

- assistant lookup from the assistants table
- graph resolution from `LangGraphService`
- auth enforcement from the current auth dependency flow
- runtime store and checkpointer wiring from the existing execution path

Avoid creating a parallel assistant model or duplicate graph invocation pipeline.

## Assistant and graph resolution

For each A2A request:

1. authenticate the caller using the same auth path as the rest of the API
2. resolve `{assistant_id}` from the assistants table
3. load the corresponding graph entry from `LangGraphService`
4. verify that the graph entry is A2A-compatible
5. execute or query task state based on the requested RPC method

Unknown assistants return a protocol-shaped error rather than a REST-shaped 404 body.

## Graph compatibility model

The A2A protocol requires message-based interaction. For this milestone, compatibility should be explicit and conservative.

Compatibility rule:

- a graph is A2A-compatible if its declared or inferred input schema supports a top-level `messages` field suitable for message payload delivery

Non-compatible graphs should:

- still exist normally for REST or MCP use
- not be callable through A2A
- produce a clear JSON-RPC error indicating that the assistant’s graph is not A2A-compatible

Do not synthesize fake `messages` wrappers for arbitrary object schemas.

## Runtime mapping

### contextId

A2A `contextId` represents conversational continuity.

Behavior:

- first turn without `contextId`: generate a new one
- follow-up turns with `contextId`: reuse it
- map the effective `contextId` into runtime execution as the effective `thread_id`

This preserves LangSmith-style tracing semantics and keeps the protocol continuity key aligned with internal execution identity.

### taskId

A2A `taskId` identifies an individual task handle.

Behavior:

- `message/send`
  - create or reuse a task record for the request
  - execute the assistant
  - return a completed result payload with task metadata
- `message/stream`
  - create a task record
  - stream task progress and final output over SSE
  - preserve the latest task snapshot for later `tasks/get`
- `tasks/get`
  - return the latest task snapshot for the provided `taskId`
- `tasks/cancel`
  - if the task is still active, request cancellation and mark the task accordingly
  - if the task is already terminal, return the terminal task state without fabricating an error

## Task registry

Use an in-process A2A task registry for this milestone.

The registry should track:

- `taskId`
- `assistant_id`
- `contextId`
- lifecycle state
- latest status message
- final output artifacts, when available
- cancellation handle or cancellation state, when applicable

Requirements:

- thread-safe or async-safe access for concurrent requests
- bounded lifetime or cleanup to avoid unbounded growth
- no durability claims across process restarts

This intentionally matches the chosen parity target rather than exceeding it.

## Request normalization

### Inbound A2A message

Normalize the A2A message payload into the graph’s expected message-shaped input:

- consume A2A text parts
- convert them into the internal `messages` structure expected by the graph
- preserve caller-provided `contextId` and `taskId` for runtime/task wiring

For the first milestone:

- support text parts as the canonical content type
- reject unsupported or malformed parts clearly

### Outbound A2A result

Normalize graph output into a strict A2A response shape:

- return text content as A2A text parts
- populate task/result metadata consistently
- keep the payload minimal and deterministic

Do not guess at rich media or non-text artifacts unless the graph output is already represented in a directly mappable form and the contract is clear.

## Streaming

`message/stream` should return a Server-Sent Events response that reflects A2A streaming behavior closely enough for client interoperability.

Requirements:

- protocol-valid SSE framing
- ordered task updates
- final terminal event
- cancellation-aware stream closure

The implementation may adapt from the existing execution stream internally, but the wire format must be A2A-shaped rather than reusing REST run-stream payloads verbatim.

## Error handling

Return protocol-shaped JSON-RPC errors for:

- malformed JSON
- malformed JSON-RPC envelope
- unsupported method
- missing required params
- unknown assistant
- unknown task
- graph incompatibility
- invalid `assistant_id` in agent-card discovery

Error behavior goals:

- clear enough for client debugging
- deterministic
- no silent coercions
- no REST-specific error body leakage

## Agent Card

`GET /.well-known/agent-card.json?assistant_id={assistant_id}` should:

- require authentication consistent with assistant access
- resolve the assistant and backing graph
- return assistant-first identity fields
- describe the A2A endpoint URL for that assistant

Card content should be derived as follows:

- identity: assistant row
- protocol endpoint: A2A route for the assistant
- capabilities and skill-like details: bound graph entry and compatibility metadata
- input/output modes: conservative reflection of the supported A2A text interaction model

Unknown or incompatible assistants should fail clearly rather than returning partial cards.

## Testing strategy

### Unit tests

- `tests/unit/test_a2a_config.py`
  - default enablement
  - explicit disablement
  - malformed config fail-closed
  - malformed `http` fail-closed
  - malformed `disable_a2a` fail-closed
- A2A compatibility helpers
  - compatible message schema accepted
  - incompatible graph rejected
- Agent Card metadata helpers
  - assistant-first name and description
  - graph-derived capability fields

### Integration tests

- `tests/integration/test_a2a_endpoint.py`
  - auth required
  - exact path behavior
  - `message/send`
  - `message/stream`
  - `tasks/get`
  - `tasks/cancel`
  - disabled route not mounted
  - incompatible assistant rejected cleanly
- `tests/integration/test_agent_card.py`
  - card discovery success
  - missing `assistant_id`
  - unknown assistant
  - disabled A2A route behavior if applicable
- extend system endpoint coverage
  - `/info.flags.a2a`
  - `/metrics?format=json` flags

### End-to-end verification

Add an e2e interoperability test using a real A2A client against the locally running server, similar in spirit to the MCP live interoperability test.

Target proof:

- authenticated client can call `message/send`
- authenticated client can use `message/stream`
- task lookup works with returned `taskId`

## CI and verification

Local verification should rely on standard Python test execution.

If any Docker-related proof becomes necessary for this tranche, use GitHub CI rather than local Docker runs.

Expected verification slice:

- relevant unit tests
- relevant integration tests
- A2A e2e/live interoperability test
- existing nearby endpoint tests to catch regressions in startup flags or auth behavior

## README updates

Update `README.md` to describe:

- A2A endpoint availability
- supported methods
- agent-card discovery URL
- assistant-first metadata behavior
- `http.disable_a2a`
- compatibility expectations for message-based graphs

The README must describe only shipped behavior, not broader aspirational LangSmith functionality.

## Implementation plan handoff

After this spec is approved, create an implementation plan that:

1. adds fail-closed A2A config handling
2. adds A2A routing and Agent Card discovery
3. implements strict assistant and graph compatibility checks
4. implements JSON-RPC method handling and streaming
5. adds task registry and cancellation support
6. adds unit, integration, and e2e coverage
7. updates README and verifies the full slice
