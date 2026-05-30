from __future__ import annotations

import asyncio
import json
import time
from uuid import uuid4

import httpx
import pytest

from tests.e2e.live_provider_helpers import (
    fetch_store_item_from_backend,
    parse_sse_events,
    provider_capability_enabled,
    provider_graph_id,
    user_headers,
)


def _text_from_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if "content" in content:
            return _text_from_content(content["content"])
        return ""
    if isinstance(content, list):
        return "".join(_text_from_content(item) for item in content)
    return ""


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


async def _poll_run(
    *,
    client: httpx.AsyncClient,
    thread_id: str,
    run_id: str,
    user_id: str,
    timeout_seconds: float = 120.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        response = await client.get(f"/threads/{thread_id}/runs/{run_id}", headers=user_headers(user_id))
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"success", "error", "interrupted"}:
            return payload
        if time.monotonic() > deadline:
            pytest.fail(f"Run {run_id} did not reach a terminal state within {timeout_seconds:.0f}s")
        await asyncio.sleep(1)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_live_provider_streaming_http_flow(live_provider_base_url: str) -> None:
    user_id = f"provider-stream-{uuid4().hex}"

    async with httpx.AsyncClient(base_url=live_provider_base_url, timeout=60.0, trust_env=False) as client:
        assistant = await client.post(
            "/assistants",
            json={"name": "live-provider-stream", "graph_id": provider_graph_id("stream")},
        )
        assert assistant.status_code == 200
        assistant_id = assistant.json()["assistant_id"]

        thread = await client.post("/threads", json={"metadata": {"suite": "live-provider-stream"}}, headers=user_headers(user_id))
        assert thread.status_code == 200
        thread_id = thread.json()["thread_id"]

        run = await client.post(
            f"/threads/{thread_id}/runs",
            json={
                "assistant_id": assistant_id,
                "input": {
                    "message": (
                        "Explain why end-to-end streaming verification matters in exactly two sentences, "
                        "using at least forty words and no bullet points."
                    )
                },
            },
            headers=user_headers(user_id),
        )
        assert run.status_code == 200
        run_id = run.json()["run_id"]

        waited_body = await _poll_run(client=client, thread_id=thread_id, run_id=run_id, user_id=user_id)
        assert waited_body["status"] == "success"
        assert waited_body["output"]["final_text"]

        fetched = await client.get(f"/threads/{thread_id}/runs/{run_id}", headers=user_headers(user_id))
        assert fetched.status_code == 200
        assert fetched.json()["output"]["final_text"] == waited_body["output"]["final_text"]

        stream = await client.get(f"/threads/{thread_id}/runs/{run_id}/stream", headers=user_headers(user_id))
        assert stream.status_code == 200
        payloads = [
            json.loads(line.replace("data: ", "", 1))
            for line in stream.text.splitlines()
            if line.startswith("data: ")
        ]
        message_chunks = [
            payload
            for payload in payloads
            if payload.get("event") == "message_chunk"
            and payload.get("langgraph_event") in {"on_chat_model_stream", "on_llm_stream"}
            and _text_from_content(payload.get("content")).strip()
        ]

        assert payloads[0]["event"] == "start"
        assert "event: start" in stream.text
        assert "event: end" in stream.text
        assert len(message_chunks) >= 2
        assert _normalize_text("".join(_text_from_content(payload.get("content")) for payload in message_chunks)) == _normalize_text(
            waited_body["output"]["final_text"]
        )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_live_provider_create_time_wait_and_stream_match_langgraph_contract(live_provider_base_url: str) -> None:
    user_id = f"provider-create-time-{uuid4().hex}"

    async with httpx.AsyncClient(base_url=live_provider_base_url, timeout=60.0, trust_env=False) as client:
        assistant = await client.post(
            "/assistants",
            json={"name": "live-provider-create-time", "graph_id": provider_graph_id("stream")},
        )
        assert assistant.status_code == 200
        assistant_id = assistant.json()["assistant_id"]

        thread = await client.post(
            "/threads",
            json={"metadata": {"suite": "live-provider-create-time"}},
            headers=user_headers(user_id),
        )
        assert thread.status_code == 200
        thread_id = thread.json()["thread_id"]

        waited_create = await client.post(
            f"/threads/{thread_id}/runs/wait",
            json={
                "assistant_id": assistant_id,
                "on_disconnect": "continue",
                "durability": "async",
                "input": {
                    "message": (
                        "Reply with one short sentence about create-time wait coverage."
                    )
                },
            },
            headers=user_headers(user_id),
        )
        assert waited_create.status_code == 200
        waited_body = waited_create.json()
        assert "run_id" not in waited_body
        assert "status" not in waited_body
        messages = waited_body.get("messages")
        assert isinstance(messages, list)
        assert len(messages) >= 2
        assert _text_from_content(messages[-1].get("content") if isinstance(messages[-1], dict) else "").strip()
        wait_run_id = waited_create.headers["content-location"].rpartition("/")[2]
        assert waited_create.headers["content-location"] == f"/threads/{thread_id}/runs/{wait_run_id}"
        assert waited_create.headers["location"] == f"/threads/{thread_id}/runs/{wait_run_id}/join"

        joined_wait = await client.get(waited_create.headers["location"], headers=user_headers(user_id))
        assert joined_wait.status_code == 200
        assert joined_wait.headers["content-location"] == waited_create.headers["content-location"]
        assert joined_wait.json() == waited_body

        streamed_create = await client.post(
            f"/threads/{thread_id}/runs/stream",
            json={
                "assistant_id": assistant_id,
                "input": {
                    "message": (
                        "Reply with one short sentence about create-time stream coverage."
                    )
                },
                "stream_mode": "updates",
            },
            headers=user_headers(user_id),
        )
        assert streamed_create.status_code == 200
        assert streamed_create.headers["content-type"].startswith("text/event-stream")
        events = parse_sse_events(streamed_create.text)
        event_names = [event["event"] for event in events]
        assert event_names[0] == "metadata"
        assert "updates" in event_names
        assert "start" not in event_names
        assert "message_chunk" not in event_names
        stream_run_id = next(str(event["data"]["run_id"]) for event in events if event["event"] == "metadata")
        assert streamed_create.headers["content-location"] == f"/threads/{thread_id}/runs/{stream_run_id}"
        assert streamed_create.headers["location"] == (
            f"/threads/{thread_id}/runs/{stream_run_id}/stream?stream_mode=updates"
        )

        joined_stream = await client.get(streamed_create.headers["location"], headers=user_headers(user_id))
        assert joined_stream.status_code == 200
        joined_events = parse_sse_events(joined_stream.text)
        joined_event_names = [event["event"] for event in joined_events]
        assert joined_event_names[0] == "metadata"
        assert "updates" in joined_event_names
        assert "start" not in joined_event_names
        assert "message_chunk" not in joined_event_names


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_live_provider_store_endpoints_and_graph(live_provider_base_url: str) -> None:
    if not provider_capability_enabled("store"):
        pytest.skip("Live provider store coverage is disabled for this backend tier.")

    user_id = f"provider-store-{uuid4().hex}"
    other_user_id = f"provider-store-other-{uuid4().hex}"
    namespace = ["live-provider", "store", uuid4().hex]
    memory_key = f"memory-{uuid4().hex}"

    async with httpx.AsyncClient(base_url=live_provider_base_url, timeout=60.0, trust_env=False) as client:
        created = await client.put(
            "/store/items",
            json={"namespace": namespace, "key": "profile", "value": {"kind": "profile", "name": "Ada"}},
            headers=user_headers(user_id),
        )
        assert created.status_code == 200

        fetched = await client.get(
            "/store/items",
            params=[("key", "profile"), *(("namespace", part) for part in namespace)],
            headers=user_headers(user_id),
        )
        assert fetched.status_code == 200
        assert fetched.json()["value"] == {"kind": "profile", "name": "Ada"}

        isolated = await client.get(
            "/store/items",
            params=[("key", "profile"), *(("namespace", part) for part in namespace)],
            headers=user_headers(other_user_id),
        )
        assert isolated.status_code == 404

        assistant = await client.post(
            "/assistants",
            json={"name": "live-provider-store", "graph_id": provider_graph_id("store_memory")},
        )
        assert assistant.status_code == 200
        assistant_id = assistant.json()["assistant_id"]

        thread = await client.post("/threads", json={"metadata": {"suite": "live-provider-store-graph"}}, headers=user_headers(user_id))
        assert thread.status_code == 200
        thread_id = thread.json()["thread_id"]

        run = await client.post(
            f"/threads/{thread_id}/runs",
            json={
                "assistant_id": assistant_id,
                "input": {
                    "memory_key": memory_key,
                    "memory_value": {"kind": "profile", "name": "Ada Lovelace"},
                },
            },
            headers=user_headers(user_id),
        )
        assert run.status_code == 200
        run_id = run.json()["run_id"]

        waited_body = await _poll_run(client=client, thread_id=thread_id, run_id=run_id, user_id=user_id)
        assert waited_body["status"] == "success"
        assert waited_body["output"]["namespace"] == ["graph", "memory"]
        assert waited_body["output"]["key"] == memory_key
        assert waited_body["output"]["value"]["name"] == "Ada Lovelace"
        assert waited_body["output"]["value"]["provider_summary"]

        backend_row = await fetch_store_item_from_backend(user_id=user_id, namespace=["graph", "memory"], key=memory_key)
        assert backend_row is not None
        assert backend_row["namespace"] == ["graph", "memory"]
        assert backend_row["value"]["name"] == "Ada Lovelace"
        assert backend_row["value"]["provider_summary"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_live_provider_store_ttl_expires_items_on_mysql_family_backend(live_provider_base_url: str) -> None:
    if not provider_capability_enabled("store"):
        pytest.skip("Live provider store coverage is disabled for this backend tier.")

    user_id = f"provider-store-ttl-{uuid4().hex}"
    namespace = ["live-provider", "ttl", uuid4().hex]

    async with httpx.AsyncClient(base_url=live_provider_base_url, timeout=30.0, trust_env=False) as client:
        created = await client.put(
            "/store/items",
            json={
                "namespace": namespace,
                "key": "ephemeral",
                "value": {"kind": "note", "name": "expires-from-live-provider-suite"},
            },
            headers=user_headers(user_id),
        )
        assert created.status_code == 200

        immediate = await client.get(
            "/store/items",
            params=[("key", "ephemeral"), *(("namespace", part) for part in namespace)],
            headers=user_headers(user_id),
        )
        assert immediate.status_code == 200

        await asyncio.sleep(4.0)

        expired = await client.get(
            "/store/items",
            params=[("key", "ephemeral"), *(("namespace", part) for part in namespace)],
            headers=user_headers(user_id),
        )
        assert expired.status_code == 404

        searched = await client.post(
            "/store/items/search",
            json={"namespace_prefix": namespace[:2], "limit": 10, "offset": 0},
            headers=user_headers(user_id),
        )
        assert searched.status_code == 200
        assert searched.json() == {"items": []}


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_live_provider_hitl_rest_and_protocol_resume(live_provider_base_url: str) -> None:
    if not provider_capability_enabled("hitl"):
        pytest.skip("Live provider HITL coverage is disabled for this backend tier.")

    user_id = f"provider-hitl-{uuid4().hex}"

    async with httpx.AsyncClient(base_url=live_provider_base_url, timeout=120.0, trust_env=False) as client:
        assistant = await client.post(
            "/assistants",
            json={"name": "live-provider-hitl", "graph_id": provider_graph_id("hitl")},
        )
        assert assistant.status_code == 200
        assistant_id = assistant.json()["assistant_id"]

        thread = await client.post("/threads", json={"metadata": {"suite": "live-provider-hitl"}}, headers=user_headers(user_id))
        assert thread.status_code == 200
        thread_id = thread.json()["thread_id"]

        run = await client.post(
            f"/threads/{thread_id}/runs",
            json={"assistant_id": assistant_id, "input": {"foo": "hello "}},
            headers=user_headers(user_id),
        )
        assert run.status_code == 200
        run_id = run.json()["run_id"]

        interrupted_stream = await client.post(
            f"/threads/{thread_id}/stream",
            json={"channels": ["lifecycle", "input", "values"]},
            headers=user_headers(user_id),
        )
        assert interrupted_stream.status_code == 200
        interrupted_events = parse_sse_events(interrupted_stream.text)
        input_event = next(event for event in interrupted_events if event["event"] == "input.requested")
        interrupt_payload = input_event["data"]["params"]["data"]
        assert interrupt_payload["payload"] == "Provide value:"
        interrupt_id = interrupt_payload["interrupt_id"]
        lifecycle_states = [event["data"]["params"]["data"]["event"] for event in interrupted_events if event["event"] == "lifecycle"]
        assert lifecycle_states == ["started", "interrupted"]

        interrupted_run = await _poll_run(client=client, thread_id=thread_id, run_id=run_id, user_id=user_id)
        assert interrupted_run["status"] == "interrupted"
        assert interrupted_run["interrupts"][0]["value"] == "Provide value:"

        last_seq = int(interrupted_events[-1]["id"])
        responded = await client.post(
            f"/threads/{thread_id}/commands",
            json={
                "id": 1,
                "method": "input.respond",
                "params": {
                    "namespace": [],
                    "interrupt_id": interrupt_id,
                    "response": "world",
                },
            },
            headers=user_headers(user_id),
        )
        assert responded.status_code == 200
        responded_body = responded.json()
        assert responded_body["type"] == "success"
        assert responded_body["id"] == 1

        resumed_stream = await client.post(
            f"/threads/{thread_id}/stream",
            json={"channels": ["lifecycle", "values"], "since": last_seq},
            headers=user_headers(user_id),
        )
        assert resumed_stream.status_code == 200
        resumed_events = parse_sse_events(resumed_stream.text)
        resumed_states = [event["data"]["params"]["data"]["event"] for event in resumed_events if event["event"] == "lifecycle"]
        assert resumed_states == ["started", "completed"]
        resumed_values = [event["data"]["params"]["data"] for event in resumed_events if event["event"] == "values"]
        assert resumed_values[-1]["foo"].endswith("world")

        final_run = await _poll_run(client=client, thread_id=thread_id, run_id=run_id, user_id=user_id)
        assert final_run["status"] == "success"
        assert final_run["output"]["state"]["foo"].endswith("world")

        create_time_thread = await client.post(
            "/threads",
            json={"metadata": {"suite": "live-provider-hitl-create-time-stream"}},
            headers=user_headers(user_id),
        )
        assert create_time_thread.status_code == 200
        create_time_thread_id = create_time_thread.json()["thread_id"]

        create_time_stream = await client.post(
            f"/threads/{create_time_thread_id}/runs/stream",
            json={"assistant_id": assistant_id, "input": {"foo": "hello "}, "stream_mode": "messages"},
            headers=user_headers(user_id),
        )
        assert create_time_stream.status_code == 200
        create_time_events = parse_sse_events(create_time_stream.text)
        create_time_input = next(event for event in create_time_events if event["event"] == "input.requested")
        assert create_time_input["data"]["payload"] == "Provide value:"
        assert create_time_input["data"]["interrupt_id"]
