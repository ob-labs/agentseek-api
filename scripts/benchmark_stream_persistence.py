"""Compare revisions through the real API using SQLite or embedded seekdb.

Run with PYTHONPATH pointing to a revision's src directory. No provider calls,
no mocked executor/checkpointer, and no replacement persistence implementation.

From a checkout with the locked development environment installed:
    PYTHONPATH=src .venv/bin/python scripts/benchmark_stream_persistence.py \
        --messages 38 --rtt-ms 10 --output /tmp/stream-fresh.json
    PYTHONPATH=src .venv/bin/python scripts/benchmark_stream_persistence.py \
        --messages 0 --history 38 --rtt-ms 100 --output /tmp/stream-history.json
    # Install the locked optional backend first: uv sync --frozen --extra embedded
    PYTHONPATH=src .venv/bin/python scripts/benchmark_stream_persistence.py \
        --backend embedded --repeat 5 --messages 38 --max-api-seconds 1 \
        --max-finalization-seconds 0.25 --output /tmp/stream-embedded.json

Embedded mode uses the application's normal SQLite metadata plus native seekdb
run/LangGraph checkpoints and store, in a temporary directory. SQLite mode
uses SQLite run checkpoints and an in-memory LangGraph checkpointer. No latency
is injected unless --rtt-ms is set. Initialization is timed separately; each
repetition uses a fresh thread. Finalization runs from execute_run returning
(including its checkpoint save) until the API response completes.
"""

import argparse
import asyncio
from collections import Counter, defaultdict
from contextvars import ContextVar
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import platform
import statistics
import tempfile
import time

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from sqlalchemy import event, text
from sqlalchemy.util.concurrency import await_only

import agentseek_api
from agentseek_api.main import create_app
from agentseek_api.core.database import db_manager
from agentseek_api.services import run_jobs, run_preparation
from agentseek_api.services.langgraph_service import get_langgraph_service
from agentseek_api.services.run_state import run_broker
from agentseek_api.services.sse import safe_json_dumps
from agentseek_api.services.stream_persistence import load_thread_stream_events
from agentseek_api.services.thread_protocol import (
    thread_protocol_broker,
    protocol_channel_for_method,
)
from agentseek_api.settings import settings

parser = argparse.ArgumentParser()
parser.add_argument("--messages", type=int, default=45)
parser.add_argument(
    "--history", type=int, default=0, help="Messages emitted by an untimed prior run"
)
parser.add_argument("--rtt-ms", type=float, default=0)
parser.add_argument("--backend", choices=["sqlite", "embedded"], default="sqlite")
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--max-api-seconds", type=float)
parser.add_argument("--max-finalization-seconds", type=float)
parser.add_argument("--max-metadata-commits", type=int)
parser.add_argument("--max-snapshot-selects", type=int)
parser.add_argument("--output", required=True)
args = parser.parse_args()
if args.repeat < 1 or min(args.messages, args.history, args.rtt_ms) < 0:
    parser.error(
        "repeat must be positive; message counts and latency must be nonnegative"
    )
phase = ContextVar("profile_phase", default="api")
calls = defaultdict(Counter)
store_calls = defaultdict(Counter)
durations = defaultdict(float)
finished_at = {}
latency = 0.0
snapshot_sizes = []
node_seconds = []


def build_graph(checkpointer=None, store=None):
    def produce(state):
        started = time.perf_counter()
        messages = [
            AIMessage(content=f"message-{i}", id=f"proof-message-{i}")
            for i in range(state["count"])
        ]
        node_seconds.append(time.perf_counter() - started)
        return {"messages": messages, "ok": True}

    graph = StateGraph(dict)
    graph.add_node("produce", produce)
    graph.add_edge(START, "produce")
    graph.add_edge("produce", END)
    return graph.compile(checkpointer=checkpointer, store=store)


def extract_output(result, _input):
    return {
        "messages": [message.model_dump(mode="json") for message in result["messages"]],
        "ok": result["ok"],
    }


def measured(name, function):
    async def call(*pos, **kwargs):
        token = phase.set(name)
        started = time.perf_counter()
        if name == "final_snapshot":
            snapshot_sizes.append(len(thread_protocol_broker.snapshot_records(pos[0])))
        try:
            return await function(*pos, **kwargs)
        finally:
            durations[name] += time.perf_counter() - started
            finished_at[name] = time.perf_counter()
            phase.reset(token)

    return call


run_jobs.execute_run = measured(
    "graph_and_stream_including_checkpoint", run_jobs.execute_run
)
run_preparation.execute_run = run_jobs.execute_run
run_jobs._persist_thread_snapshot = measured(
    "final_snapshot", run_jobs._persist_thread_snapshot
)
run_preparation._persist_thread_snapshot = run_jobs._persist_thread_snapshot
db_manager.run_checkpointer_call = measured(
    "run_checkpoint", db_manager.run_checkpointer_call
)


def on_sql(_connection, _cursor, statement, _parameters, _context, _executemany):
    calls[phase.get()][statement.split()[0].upper()] += 1
    store_calls["metadata"][statement.split()[0].upper()] += 1
    if latency:
        # Yield inside SQLAlchemy's async greenlet, so polling and other tasks
        # continue just as they would while waiting for a remote SQL response.
        await_only(asyncio.sleep(latency))


def on_commit(_connection):
    calls[phase.get()]["COMMIT"] += 1
    store_calls["metadata"]["COMMIT"] += 1
    if latency:
        await_only(asyncio.sleep(latency))


def checkpoint_sql(
    _connection, _cursor, statement, _parameters, _context, _executemany
):
    calls[phase.get()][statement.split()[0].upper()] += 1
    store_calls["run_checkpoint"][statement.split()[0].upper()] += 1
    if latency:
        time.sleep(latency)  # This synchronous checkpointer runs in a worker thread.


def checkpoint_commit(_connection):
    calls[phase.get()]["COMMIT"] += 1
    store_calls["run_checkpoint"]["COMMIT"] += 1
    if latency:
        time.sleep(latency)


def langgraph_sql(_connection, _cursor, statement, _parameters, _context, _many):
    store_calls["langgraph_checkpoint"][statement.split()[0].upper()] += 1
    if latency:
        time.sleep(latency)


def langgraph_commit(_connection):
    store_calls["langgraph_checkpoint"]["COMMIT"] += 1
    if latency:
        time.sleep(latency)


def benchmark(client):
    global latency
    assistant = client.post("/assistants", json={"name": "proof", "graph_id": "proof"})
    assert assistant.status_code == 200, assistant.text
    thread = client.post("/threads", json={})
    assert thread.status_code == 200, thread.text
    thread_id = thread.json()["thread_id"]
    endpoint = f"/threads/{thread_id}/runs/wait"
    assistant_id = assistant.json()["assistant_id"]
    if args.history:
        seed = client.post(
            endpoint,
            json={"assistant_id": assistant_id, "input": {"count": args.history}},
        )
        assert (
            seed.status_code == 200 and len(seed.json()["messages"]) == args.history
        ), seed.text
    prior_events = len(thread_protocol_broker.snapshot_records(thread_id))
    calls.clear()
    store_calls.clear()
    durations.clear()
    finished_at.clear()
    node_seconds.clear()
    snapshot_sizes.clear()
    engine = db_manager.get_engine().sync_engine
    event.listen(engine, "before_cursor_execute", on_sql)
    event.listen(engine, "commit", on_commit)
    checkpoint_engine = db_manager.get_checkpointer()._engine
    event.listen(checkpoint_engine, "before_cursor_execute", checkpoint_sql)
    event.listen(checkpoint_engine, "commit", checkpoint_commit)
    graph_saver = db_manager.get_langgraph_checkpointer()
    graph_engine = getattr(getattr(graph_saver, "obvector", None), "engine", None)
    if graph_engine is not None:
        event.listen(graph_engine, "before_cursor_execute", langgraph_sql)
        event.listen(graph_engine, "commit", langgraph_commit)
    latency = args.rtt_ms / 1000
    started = time.perf_counter()
    response = client.post(
        endpoint,
        json={"assistant_id": assistant_id, "input": {"count": args.messages}},
    )
    response_finished = time.perf_counter()
    elapsed = response_finished - started
    finalization = (
        response_finished - finished_at["graph_and_stream_including_checkpoint"]
    )
    latency = 0
    event.remove(engine, "before_cursor_execute", on_sql)
    event.remove(engine, "commit", on_commit)
    event.remove(checkpoint_engine, "before_cursor_execute", checkpoint_sql)
    event.remove(checkpoint_engine, "commit", checkpoint_commit)
    if graph_engine is not None:
        event.remove(graph_engine, "before_cursor_execute", langgraph_sql)
        event.remove(graph_engine, "commit", langgraph_commit)
    assert response.status_code == 200, response.text
    output = response.json()
    expected_content = [f"message-{i}" for i in range(args.messages)]
    assert [message["content"] for message in output["messages"]] == expected_content, (
        output
    )
    run_id = response.headers["content-location"].split("/")[-1]
    run = client.get(f"/threads/{thread_id}/runs/{run_id}")
    assert run.status_code == 200 and run.json()["status"] == "success", run.text
    with checkpoint_engine.connect() as connection:
        checkpoint = json.loads(
            connection.scalar(
                text(
                    "SELECT checkpoint FROM agentseek_checkpoints WHERE run_id=:run_id"
                ),
                {"run_id": run_id},
            )
        )
    assert [
        message["content"] for message in checkpoint["output"]["messages"]
    ] == expected_content
    # Read the real LangGraph saver too: embedded mode must not silently use
    # the SQLite-mode InMemorySaver or merely save a completed-run summary.
    config = {"configurable": {"thread_id": thread_id}}
    graph_checkpoint = graph_saver.get_tuple(config)
    assert graph_checkpoint is not None
    graph_values = graph_checkpoint.checkpoint["channel_values"]["__root__"]
    assert [item.content for item in graph_values["messages"]] == expected_content
    seekdb_version = None
    if args.backend == "embedded":
        assert "path" in db_manager.get_checkpointer().connection_args
        assert type(graph_saver).__module__ == "langchain_oceanbase.checkpointer"
        with graph_engine.connect() as connection:
            seekdb_version = connection.scalar(text("SELECT version()"))
            checkpoint_count = connection.scalar(
                text("SELECT COUNT(*) FROM checkpoints WHERE thread_id=:thread_id"),
                {"thread_id": thread_id},
            )
        assert checkpoint_count > 0
    buffered = thread_protocol_broker.snapshot_records(thread_id)
    channels = list({protocol_channel_for_method(item["method"]) for item in buffered})
    thread_protocol_broker.delete_thread(thread_id)

    async def load_history():
        return await load_thread_stream_events(
            thread_id, channels=channels, namespaces=None, depth=None
        )

    persisted = client.portal.call(load_history)
    assert persisted == buffered, (len(persisted), len(buffered))
    expected_run_events = [
        (seq, json.loads(safe_json_dumps({"run_id": run_id, **payload})))
        for seq, payload in run_broker.snapshot_records(run_id)
    ]
    run_broker._events.pop(run_id, None)
    replay = client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
    assert replay.status_code == 200 and "event: end\n" in replay.text, replay.text
    replayed_run_events = []
    for frame in replay.text.split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in frame.splitlines() if ": " in line
        )
        if "data" in fields:
            replayed_run_events.append((int(fields["id"]), json.loads(fields["data"])))
    assert replayed_run_events == expected_run_events
    report = {
        "source": agentseek_api.__file__,
        "backend": {
            "mode": args.backend,
            "metadata_dialect": engine.dialect.name,
            "run_checkpoint_class": type(db_manager.get_checkpointer()).__module__,
            "langgraph_checkpoint_class": type(graph_saver).__module__,
            "seekdb_version": seekdb_version,
            "pylibseekdb_version": version("pylibseekdb")
            if args.backend == "embedded"
            else None,
            "platform": platform.platform(),
        },
        "scenario": {
            "message_count": args.messages,
            "history_message_count": args.history,
            "injected_rtt_ms_per_sql_or_commit": args.rtt_ms,
        },
        "prior_thread_events": prior_events,
        "snapshot_event_counts": list(snapshot_sizes),
        "api_elapsed_seconds": round(elapsed, 4),
        "framework_seconds": round(elapsed - sum(node_seconds), 4),
        "finalization_seconds": round(finalization, 4),
        "graph_node_seconds": round(sum(node_seconds), 6),
        "phase_seconds": {key: round(value, 4) for key, value in durations.items()},
        "database_calls_by_phase": dict(calls),
        "database_calls_by_store": {
            name: dict(counts) for name, counts in store_calls.items()
        },
        "metadata_commits": store_calls["metadata"]["COMMIT"],
        "snapshot_selects": calls["final_snapshot"]["SELECT"],
        "checks": {
            "status": "success",
            "output_matches_expected": True,
            "run_checkpoint_saved": True,
            "langgraph_checkpoint_saved": True,
            "thread_events_replayed_after_memory_clear": len(persisted),
            "run_events_replayed_after_memory_clear": len(replayed_run_events),
            "run_stream_replays_terminal_event_after_memory_clear": True,
        },
        "output_content_sha256": hashlib.sha256(
            json.dumps(expected_content).encode()
        ).hexdigest(),
    }
    return report


with tempfile.TemporaryDirectory() as directory:
    settings.METADATA_DB_URL = f"sqlite+aiosqlite:///{directory}/proof.db"
    settings.METADATA_DB_BACKEND = "sqlite"
    settings.SEEKDB_EMBED = args.backend == "embedded"
    settings.SEEKDB_EMBED_DIR = f"{directory}/seekdb"
    settings.OCEANBASE_DB_NAME = "stream_proof"
    settings.EXECUTOR_BACKEND = "inline"
    settings.AGENTSEEK_GRAPHS = None
    settings.AUTH_MODULE_PATH = None
    get_langgraph_service().register(
        "proof",
        graph_factory=build_graph,
        prepare_input=lambda value: value,
        extract_output=extract_output,
    )
    startup_started = time.perf_counter()
    with TestClient(create_app()) as client:
        startup_seconds = time.perf_counter() - startup_started
        reports = [benchmark(client) for _ in range(args.repeat)]
        # Preserve the original single-run report fields for existing consumers.
        report = dict(reports[0])
        report["startup_seconds"] = round(startup_seconds, 4)
        report["repetitions"] = args.repeat
        report["samples"] = reports
        report["summary"] = {
            metric: {
                "median": round(statistics.median(item[metric] for item in reports), 4),
                "max": max(item[metric] for item in reports),
            }
            for metric in (
                "api_elapsed_seconds",
                "framework_seconds",
                "finalization_seconds",
                "metadata_commits",
                "snapshot_selects",
            )
        }
        report["performance_limits"] = {
            "api_seconds": args.max_api_seconds,
            "finalization_seconds": args.max_finalization_seconds,
            "metadata_commits": args.max_metadata_commits,
            "snapshot_selects": args.max_snapshot_selects,
        }
        violations = []
        for metric, limit in (
            ("api_elapsed_seconds", args.max_api_seconds),
            ("finalization_seconds", args.max_finalization_seconds),
            ("metadata_commits", args.max_metadata_commits),
            ("snapshot_selects", args.max_snapshot_selects),
        ):
            if limit is not None and report["summary"][metric]["max"] > limit:
                violations.append(f"{metric} exceeded {limit}")
        report["performance_violations"] = violations
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if violations:
            raise SystemExit("; ".join(violations))
