"""Compare revisions through the real API with simulated SQL round trips.

Run with PYTHONPATH pointing to a revision's src directory. No provider calls,
no mocked executor/checkpointer, and no replacement persistence implementation.

From a checkout with the locked development environment installed:
    PYTHONPATH=src .venv/bin/python scripts/benchmark_stream_persistence.py \
        --messages 38 --rtt-ms 10 --output /tmp/stream-fresh.json
    PYTHONPATH=src .venv/bin/python scripts/benchmark_stream_persistence.py \
        --messages 0 --history 38 --rtt-ms 100 --output /tmp/stream-history.json

Metadata and completed-run checkpoints use temporary SQLite databases. The
normal SQLite-mode LangGraph checkpointer is in memory. Timing is synthetic;
compare SQL call counts and correctness checks as well as wall-clock results.
"""

import argparse
import asyncio
from collections import Counter, defaultdict
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
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
parser.add_argument("--output", required=True)
args = parser.parse_args()
phase = ContextVar("profile_phase", default="api")
calls = defaultdict(Counter)
durations = defaultdict(float)
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
    if latency:
        # Yield inside SQLAlchemy's async greenlet, so polling and other tasks
        # continue just as they would while waiting for a remote SQL response.
        await_only(asyncio.sleep(latency))


def on_commit(_connection):
    calls[phase.get()]["COMMIT"] += 1
    if latency:
        await_only(asyncio.sleep(latency))


def checkpoint_sql(
    _connection, _cursor, statement, _parameters, _context, _executemany
):
    calls[phase.get()][statement.split()[0].upper()] += 1
    if latency:
        time.sleep(latency)  # This synchronous checkpointer runs in a worker thread.


def checkpoint_commit(_connection):
    calls[phase.get()]["COMMIT"] += 1
    if latency:
        time.sleep(latency)


with tempfile.TemporaryDirectory() as directory:
    settings.METADATA_DB_URL = f"sqlite+aiosqlite:///{directory}/proof.db"
    settings.METADATA_DB_BACKEND = "sqlite"
    settings.SEEKDB_EMBED = False
    settings.EXECUTOR_BACKEND = "inline"
    settings.AGENTSEEK_GRAPHS = None
    settings.AUTH_MODULE_PATH = None
    get_langgraph_service().register(
        "proof",
        graph_factory=build_graph,
        prepare_input=lambda value: value,
        extract_output=extract_output,
    )
    with TestClient(create_app()) as client:
        assistant = client.post(
            "/assistants", json={"name": "proof", "graph_id": "proof"}
        )
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
        durations.clear()
        node_seconds.clear()
        snapshot_sizes.clear()
        engine = db_manager.get_engine().sync_engine
        event.listen(engine, "before_cursor_execute", on_sql)
        event.listen(engine, "commit", on_commit)
        checkpoint_engine = db_manager.get_checkpointer()._engine
        event.listen(checkpoint_engine, "before_cursor_execute", checkpoint_sql)
        event.listen(checkpoint_engine, "commit", checkpoint_commit)
        latency = args.rtt_ms / 1000
        started = time.perf_counter()
        response = client.post(
            endpoint,
            json={"assistant_id": assistant_id, "input": {"count": args.messages}},
        )
        elapsed = time.perf_counter() - started
        latency = 0
        event.remove(engine, "before_cursor_execute", on_sql)
        event.remove(engine, "commit", on_commit)
        event.remove(checkpoint_engine, "before_cursor_execute", checkpoint_sql)
        event.remove(checkpoint_engine, "commit", checkpoint_commit)
        assert response.status_code == 200, response.text
        output = response.json()
        expected_content = [f"message-{i}" for i in range(args.messages)]
        assert [
            message["content"] for message in output["messages"]
        ] == expected_content, output
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
        buffered = thread_protocol_broker.snapshot_records(thread_id)
        channels = list(
            {protocol_channel_for_method(item["method"]) for item in buffered}
        )
        thread_protocol_broker.delete_thread(thread_id)

        async def load_history():
            return await load_thread_stream_events(
                thread_id, channels=channels, namespaces=None, depth=None
            )

        persisted = client.portal.call(load_history)
        assert persisted == buffered, (len(persisted), len(buffered))
        run_broker._events.pop(run_id, None)
        replay = client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
        assert replay.status_code == 200 and "event: end\n" in replay.text, replay.text
        report = {
            "source": agentseek_api.__file__,
            "scenario": {
                "message_count": args.messages,
                "history_message_count": args.history,
                "injected_rtt_ms_per_sql_or_commit": args.rtt_ms,
            },
            "prior_thread_events": prior_events,
            "snapshot_event_counts": snapshot_sizes,
            "api_elapsed_seconds": round(elapsed, 4),
            "graph_node_seconds": round(sum(node_seconds), 6),
            "phase_seconds": {key: round(value, 4) for key, value in durations.items()},
            "database_calls_by_phase": dict(calls),
            "checks": {
                "status": "success",
                "output_matches_expected": True,
                "run_checkpoint_saved": True,
                "thread_events_replayed_after_memory_clear": len(persisted),
                "run_stream_replays_terminal_event_after_memory_clear": True,
            },
            "output_content_sha256": hashlib.sha256(
                json.dumps(expected_content).encode()
            ).hexdigest(),
        }
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
