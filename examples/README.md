# Sample Graph Apps

This folder contains small, self-contained LangGraph apps you can register
with agentseek-api. They're designed as copy-paste starters: pick the one
closest to what you're building, copy the subdirectory into your own
project, wire it into your registry, and you're running.

## Included samples

These graph IDs are available from the bundled sample registry:

| graph_id             | file                                   | what it shows                                              |
| -------------------- | -------------------------------------- | ---------------------------------------------------------- |
| `stress_test`        | `graphs/stress_test/graph.py`          | Deterministic loop, no LLM. Good load-test baseline.       |
| `subgraph_agent`     | `graphs/subgraph_agent/graph.py`       | Outer router delegating to a compiled inner graph.         |
| `react_agent`        | `graphs/react_agent/graph.py`          | Tool-calling ReAct loop (uses a fake chat model, offline). |
| `stress_tool_agent`  | `graphs/stress_tool_agent/graph.py`    | Sequential tool-calling stress loop, offline and repeatable. |
| `subgraph_hitl_agent`| `graphs/subgraph_hitl_agent/graph.py`  | Nested subgraph + `interrupt()` human-in-the-loop pattern. |

This sample is registered by its own manifest:

| graph_id         | file                      | what it shows                               |
| ---------------- | ------------------------- | ------------------------------------------- |
| `external_hello` | `external_graph/graph.py` | Manifest-registered external graph example. |

These directories are config-file starters. Their graph IDs become available
when you run the server with the example `agentseek.json` in that directory:

| graph_id             | file                                   | what it shows                                              |
| -------------------- | -------------------------------------- | ---------------------------------------------------------- |
| `chat`               | `minimal_agentseek/graph.py`           | Minimal `agentseek.json` starter for first-time users.      |
| `assistant_config`   | `assistant_config/graph.py`            | Config/context/metadata starter for assistant examples.     |

See `src/agentseek_api/services/sample_graphs.py` for how bundled samples are
registered and how their input / output is adapted to the API's JSON contract.

## Running them in-process

```bash
uv run python examples/run_sample_graphs.py
uv run python examples/external_graph/run.py
```

Invokes every sample graph directly through LangGraph — no HTTP server, no
SeekDB. Useful during development when you want a tight feedback loop.

## Running them through the HTTP API

Start the server:

```bash
uv run agentseek-api dev --config examples/sample_graphs_manifest.json --no-reload --port 2024
```

Create an assistant bound to the sample you want, submit a run, and wait
for the result:

```bash
curl -s -X POST http://127.0.0.1:2024/assistants \
  -H 'x-user-id: dev' -H 'content-type: application/json' \
  -d '{"name": "stress", "graph_id": "stress_test"}'

curl -s -X POST http://127.0.0.1:2024/threads \
  -H 'x-user-id: dev' -H 'content-type: application/json' \
  -d '{"metadata": {}}'

# plug the ids into the run submission:
curl -s -X POST http://127.0.0.1:2024/threads/$THREAD_ID/runs \
  -H 'x-user-id: dev' -H 'content-type: application/json' \
  -d '{"assistant_id": "'$ASSISTANT_ID'", "input": {"delay": 0.01, "steps": 2}}'

curl -s http://127.0.0.1:2024/threads/$THREAD_ID/runs/$RUN_ID/wait \
  -H 'x-user-id: dev'
```

`tests/e2e/e2e_live_http_multi_graph.py` does the full dance against every
registered sample. `make test-cli-dev-samples` runs that sweep through the
`agentseek dev` CLI, and `make test-seekdb` runs it end-to-end against a real
SeekDB-compatible backend.

## Adding your own graph

Preferred path for this sprint:

1. Build a Python module that exposes `build_graph(checkpointer=None)`.
2. Create a JSON manifest and point `AGENTSEEK_GRAPHS` at it. Both of these
   forms are accepted:

```json
{
  "graphs": {
    "external_hello": {
      "graph": "examples.external_graph.graph:build_graph"
    }
  }
}
```

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "external_hello": "./examples/external_graph/graph.py:graph"
  }
}
```

3. Optional: add `prepare_input` / `extract_output` dotted paths if your graph
   does not use the default messages-style adapters.
4. Restart the server. User-defined manifest entries override bundled graph IDs
   on conflict.

See `examples/external_graph/manifest.json` and
`examples/external_graph/run.py` for a minimal end-to-end example.

## Auth and app mounting examples

- `examples/auth/custom_backend.py` shows an `AUTH_TYPE=custom` backend object.
- `examples/auth/jwt.md` documents the `AUTH_TYPE=jwt` environment contract.
- `examples/custom_routes/app.py` shows how to mount custom routes around
  `agentseek_api.main.create_app()`.
