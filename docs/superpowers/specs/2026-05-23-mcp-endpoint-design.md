# MCP Endpoint Design

Date: 2026-05-23
Repo: `/Users/haili/.codex/worktrees/04dd/agentseek-api`
Status: Approved for planning

## Goal

Implement strict LangGraph Agent Server style MCP endpoint support in AgentSeek
API by adding a stateless Streamable HTTP `/mcp` surface that exposes
registered graphs as MCP tools.

This work targets server-side MCP exposure only. It does not include AgentSeek
acting as an MCP client for external tools.

## Why Now

The current runtime already covers the main execution substrate needed for MCP:

- manifest-driven graph registration
- authenticated HTTP request handling
- graph execution through the existing runtime
- thread/run/state/history/event APIs

The current compatibility gap is that the runtime still advertises
`"mcp": false` in `/info` and exposes no `/mcp` endpoint. Upstream LangGraph
Agent Server documents `/mcp` as a first-class compatibility surface, so MCP is
the next meaningful parity milestone.

## Scope

### In scope

- Add a strict `/mcp` HTTP endpoint implemented inside AgentSeek
- Use Streamable HTTP transport semantics
- Keep request handling stateless
- Expose registered graphs as MCP tools
- Reuse existing API authentication behavior
- Add config support for `http.disable_mcp`
- Flip `/info.flags.mcp` to reflect actual MCP availability
- Add integration and end-to-end coverage for client interoperability
- Document the MCP endpoint in `README.md`

### Out of scope

- AgentSeek consuming external MCP servers
- Sessionful MCP behavior
- Real-provider MCP CI coverage
- Broad schema inference beyond safe, explicit behavior needed for strict v1

## Requirements

### External compatibility target

AgentSeek should behave like LangGraph Agent Server for the `/mcp` endpoint as
closely as practical within the current runtime architecture:

- endpoint path is `/mcp`
- transport is Streamable HTTP
- endpoint is stateless
- endpoint uses the same auth model as the rest of the API
- deployed graphs are automatically exposed as MCP tools
- config can disable MCP through `http.disable_mcp`

### Internal constraints

- Do not replace the existing FastAPI runtime with `langgraph-api`
- Do not introduce a second graph execution engine for MCP requests
- Reuse the current graph registry and auth stack
- Preserve existing API behavior for non-MCP routes

## Chosen Approach

Implement a native MCP router inside AgentSeek and map it onto the existing
graph registry, auth, and execution layers.

This is preferred over embedding or proxying upstream internals because this
repo is a custom FastAPI runtime rather than a thin wrapper around
`langgraph-api`. A native router keeps the integration surface explicit and
testable.

## Architecture

### High-level flow

1. Client sends a Streamable HTTP request to `/mcp`.
2. FastAPI authenticates the request using the existing auth dependency.
3. MCP router parses the request envelope and dispatches the MCP method.
4. Tool discovery reads from the registered graph manifest.
5. Tool execution maps the MCP tool call to the existing graph runtime.
6. The router returns strict MCP-compatible transport and method responses.

### Main components

#### MCP router

Add a dedicated API router, likely `src/agentseek_api/api/mcp.py`, responsible
for:

- transport request parsing
- method dispatch
- MCP-shaped success responses
- MCP-shaped protocol and method errors

#### MCP config loader

Extend config parsing so both `agentseek.json` and `langgraph.json` can carry:

```json
{
  "http": {
    "disable_mcp": true
  }
}
```

This config must influence:

- whether the `/mcp` route is registered
- whether `/info.flags.mcp` is `true` or `false`

#### Graph metadata layer

Extend manifest graph registration to preserve MCP-visible fields:

- graph id
- optional display name
- optional description
- optional explicit input schema
- optional explicit output schema

The registry should remain the source of truth for MCP tool discovery. MCP
should not depend on persisted assistants to decide what tools exist.

#### MCP tool adapter

Add an adapter that converts a registered graph entry into MCP tool metadata:

- MCP tool name
- MCP tool description
- MCP input schema

For v1, output formatting should also be normalized into MCP tool result
content.

#### Execution bridge

Add a narrow execution path that runs a graph for MCP tool calls using the same
underlying runtime behavior used by the existing API:

- resolve graph by tool name
- pass MCP tool arguments as the graph input payload
- inject authenticated user context as needed through the current config/auth
  path
- execute the graph
- normalize output into MCP tool result content

This must not create a second execution engine with divergent semantics.

## Data Model Changes

### Manifest graph entries

Current manifest entries retain execution hooks but not enough MCP metadata.
They should be extended to support object form such as:

```json
{
  "graphs": {
    "docs_agent": {
      "graph": "./docs_agent.py:graph",
      "name": "docs_agent",
      "description": "Answers repository documentation questions",
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

### Defaults

When metadata is omitted:

- tool name defaults to graph id
- description defaults to an empty string
- input schema defaults to a conservative object schema
- output schema defaults to a conservative object schema

The default schema fallback is acceptable only if the shape remains MCP-valid.
Strict parity does not require perfect inference, but it does require a stable,
explicit tool contract.

## Request and Response Mapping

### Tool discovery

MCP tool listing should enumerate registered graphs and map each one to a tool.

Each tool should include:

- name
- description
- input schema

The exact JSON structure must follow the MCP Streamable HTTP contract expected
by current clients.

### Tool invocation

Tool calls should map as follows:

- MCP tool name -> registered graph id or configured graph name
- MCP arguments -> graph input payload
- authenticated request -> current user/auth context
- graph result -> MCP tool result content

The execution bridge should prefer direct graph invocation over creating
synthetic assistant/thread rows unless protocol compatibility requires
additional wrapper behavior.

### Stateless behavior

Each `/mcp` request is processed independently. The implementation should not
introduce server-side session state for MCP connections.

## Authentication

The `/mcp` endpoint should use the same authentication model as the rest of the
AgentSeek API.

Rules:

- unauthenticated HTTP requests fail at the HTTP layer, consistent with the
  existing API
- authenticated requests proceed to MCP parsing and method dispatch
- no separate MCP-only auth stack is introduced

This keeps MCP compatibility aligned with the current auth middleware and
matches the documented upstream behavior.

## Error Handling

### HTTP-layer failures

Use standard HTTP errors for:

- missing or invalid authentication
- unsupported content type
- disabled endpoint when `/mcp` is not available

### MCP-layer failures

Use strict MCP-shaped errors for:

- malformed transport payloads
- unknown MCP methods
- unknown tool names
- invalid tool arguments
- execution failures that occur after successful MCP request parsing

Do not return ad hoc AgentSeek-native error shapes from the MCP route.

## Testing Strategy

### Unit and integration coverage

Add tests for:

- config parsing of `http.disable_mcp`
- `/info.flags.mcp` reporting
- route registration or disable behavior
- tool listing from manifest-backed graph entries
- metadata propagation for name, description, and schemas
- unknown tool handling
- auth enforcement
- graph execution through MCP tool calls

### End-to-end compatibility test

Add at least one end-to-end smoke test using an MCP client library against a
locally running AgentSeek instance at `http://127.0.0.1:2024/mcp`.

This test should prove:

- client can connect
- client can list tools
- client can invoke a tool successfully

This is required because strict parity is about client interoperability, not
just internal route shape.

### CI

Use local and integration coverage in the default CI path.

Do not move real provider-backed streaming checks into always-on CI. The manual
workflow remains the canonical proof path for real provider token streaming.

## README Updates

Document:

- `/mcp` availability
- stateless behavior
- auth expectations
- `http.disable_mcp`
- minimal client examples
- graph metadata expectations for clean MCP tool exposure

The README should distinguish:

- exposing AgentSeek graphs as MCP tools
- consuming external MCP tools from within AgentSeek graphs

Only the first is part of this milestone.

## Rollout Plan

1. Add config and registry metadata support
2. Add MCP router and request dispatch
3. Add tool discovery mapping
4. Add tool execution bridge
5. Add unit and integration tests
6. Add client interoperability smoke test
7. Update README
8. Flip `/info.flags.mcp` based on actual enablement

## Risks and Mitigations

### Risk: faux compatibility

If `/mcp` exists but does not behave like current MCP clients expect, the
feature will look complete while remaining unusable.

Mitigation:

- use an actual MCP client in end-to-end tests
- keep transport and response shapes strict

### Risk: metadata is too weak

Current graph registration does not preserve enough descriptive or schema
metadata for good tool exposure.

Mitigation:

- extend manifest graph objects explicitly
- keep defaults conservative and valid

### Risk: divergent execution semantics

If MCP tool calls use a different execution path than the existing runtime,
behavior will drift over time.

Mitigation:

- route MCP execution through the existing graph runtime
- avoid a second engine or duplicate orchestration logic

## Open Decisions Resolved

- Compatibility target: strict LangGraph Agent Server style `/mcp`
- Transport model: stateless Streamable HTTP
- Discovery source: registered graphs, not persisted assistants
- Auth model: same as the rest of the API
- Scope boundary: expose AgentSeek graphs as MCP tools only

## Acceptance Criteria

- AgentSeek exposes `/mcp` when MCP is enabled
- `/mcp` is disabled when `http.disable_mcp` is `true`
- `/info.flags.mcp` accurately reflects enabled state
- registered graphs appear as MCP tools with stable metadata
- MCP clients can list tools and invoke at least one tool successfully
- MCP requests use existing API authentication behavior
- README documents the feature and disable mechanism
