# Sample Graph Apps

This folder contains small, self-contained LangGraph apps you can register
with agentseek-api. They're designed as copy-paste starters: pick the one
closest to what you're building, copy the subdirectory into your own
project, wire it into your registry, and you're running.

## Included samples

Every sample is keyed in the API registry by its directory name:

| graph_id             | file                                   | what it shows                                              |
| -------------------- | -------------------------------------- | ---------------------------------------------------------- |
| `stress_test`        | `graphs/stress_test/graph.py`          | Deterministic loop, no LLM. Good load-test baseline.       |
| `subgraph_agent`     | `graphs/subgraph_agent/graph.py`       | Outer router delegating to a compiled inner graph.         |
| `react_agent`        | `graphs/react_agent/graph.py`          | Tool-calling ReAct loop (uses a fake chat model, offline). |
| `subgraph_hitl_agent`| `graphs/subgraph_hitl_agent/graph.py`  | Nested subgraph + `interrupt()` human-in-the-loop pattern. |

See `src/agentseek_api/services/sample_graphs.py` for how each graph is
registered and how its input / output is adapted to the API's JSON contract.

## Running them in-process

```bash
uv run python examples/run_sample_graphs.py
```

Invokes every sample graph directly through LangGraph — no HTTP server, no
SeekDB. Useful during development when you want a tight feedback loop.

## Running them through the HTTP API

Start the server:

```bash
uv run uvicorn agentseek_api.main:app --reload --port 2026
```

Create an assistant bound to the sample you want, submit a run, and wait
for the result:

```bash
curl -s -X POST http://127.0.0.1:2026/assistants \
  -H 'x-user-id: dev' -H 'content-type: application/json' \
  -d '{"name": "stress", "graph_id": "stress_test"}'

curl -s -X POST http://127.0.0.1:2026/threads \
  -H 'x-user-id: dev' -H 'content-type: application/json' \
  -d '{"metadata": {}}'

# plug the ids into the run submission:
curl -s -X POST http://127.0.0.1:2026/threads/$THREAD_ID/runs \
  -H 'x-user-id: dev' -H 'content-type: application/json' \
  -d '{"assistant_id": "'$ASSISTANT_ID'", "input": {"delay": 0.01, "steps": 2}}'

curl -s http://127.0.0.1:2026/threads/$THREAD_ID/runs/$RUN_ID/wait \
  -H 'x-user-id: dev'
```

`tests/e2e/e2e_live_http_multi_graph.py` does the full dance against every
registered sample, and `make test-seekdb` runs it end-to-end against an
embedded SeekDB instance.

## Adding your own graph

1. Drop a `graph.py` under `examples/graphs/<your_name>/` that exposes a
   compiled `graph` variable.
2. Append an entry to `build_sample_registry()` in
   `src/agentseek_api/services/sample_graphs.py` — set `graph_id`, the
   graph, and two small adapter callables (`prepare_input`, `extract_output`).
3. Restart the server. Any assistant created with `graph_id: "<your_name>"`
   now routes runs to your graph and checkpoints through the configured
   SeekDB/OceanBase backend.
