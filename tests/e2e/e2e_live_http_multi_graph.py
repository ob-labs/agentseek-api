"""End-to-end proof for every bundled sample graph.

For each sample we create an Assistant bound to the matching ``graph_id``,
submit a run with a shape the graph understands, wait for completion, and
assert on the persisted output. Drives a live uvicorn server; the caller is
responsible for pointing ``EXAMPLE_BASE_URL`` at a server that is already
talking to a real SeekDB/OceanBase backend (or a compatible MySQL).
"""

from __future__ import annotations

import json
import os

import httpx


def _run_case(
    client: httpx.Client,
    *,
    graph_id: str,
    input_payload: dict,
    expected_status: str = "success",
    assert_output,
) -> None:
    assistant = client.post("/assistants", json={"name": f"sample-{graph_id}", "graph_id": graph_id})
    assistant.raise_for_status()
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"sample": graph_id}})
    thread.raise_for_status()
    thread_id = thread.json()["thread_id"]

    run = client.post(
        f"/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": input_payload},
    )
    run.raise_for_status()
    run_id = run.json()["run_id"]

    waited = client.get(f"/threads/{thread_id}/runs/{run_id}/wait")
    waited.raise_for_status()
    body = waited.json()
    assert body["status"] == expected_status, f"{graph_id} run did not reach expected status {expected_status!r}: {body}"

    output = body.get("output") or {}
    assert_output(output)

    stream = client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
    stream.raise_for_status()
    assert "event: end" in stream.text, f"{graph_id} stream missing end event"
    print(f"{graph_id}: ok -> {json.dumps(output)[:200]}")


def _assert_stress_test(output: dict) -> None:
    assert output["final_json"]["status"] == "completed"
    assert output["final_json"]["steps_completed"] == 2


def _assert_subgraph_agent(output: dict) -> None:
    assert output["final_json"]["status"] == "completed"
    assert output["final_json"]["steps_completed"] >= 1


def _assert_react_agent(output: dict) -> None:
    transcript = output["transcript"]
    kinds = [m["type"] for m in transcript]
    assert "ToolMessage" in kinds, f"react transcript missing ToolMessage: {kinds}"
    assert "42" in output["final_text"], f"react final_text missing 42: {output['final_text']}"


def _assert_stress_tool_agent(output: dict) -> None:
    transcript = output["transcript"]
    tool_messages = [message for message in transcript if message["type"] == "ToolMessage"]
    assert len(tool_messages) == 3, f"expected 3 tool messages, got {len(tool_messages)}: {transcript}"
    assert output["final_json"]["status"] == "completed"
    assert output["final_json"]["steps_completed"] == 3


def _assert_subgraph_hitl(output: dict) -> None:
    assert output["interrupted"] is True
    assert output["interrupts"], "expected at least one interrupt"
    assert output["interrupts"][0]["value"] == "Provide value:"


def main() -> None:
    base_url = os.getenv("EXAMPLE_BASE_URL", "http://127.0.0.1:2024")
    user_id = os.getenv("EXAMPLE_USER_ID", "sample-user")
    headers = {"x-user-id": user_id}

    with httpx.Client(base_url=base_url, timeout=60.0, headers=headers) as client:
        _run_case(
            client,
            graph_id="stress_test",
            input_payload={"delay": 0.01, "steps": 2},
            assert_output=_assert_stress_test,
        )
        _run_case(
            client,
            graph_id="subgraph_agent",
            input_payload={"messages": [{"role": "user", "content": json.dumps({"delay": 0.0, "steps": 1})}]},
            assert_output=_assert_subgraph_agent,
        )
        _run_case(
            client,
            graph_id="react_agent",
            input_payload={"message": "what is the meaning of life?"},
            assert_output=_assert_react_agent,
        )
        _run_case(
            client,
            graph_id="stress_tool_agent",
            input_payload={"delay": 0.0, "steps": 3},
            assert_output=_assert_stress_tool_agent,
        )
        _run_case(
            client,
            graph_id="subgraph_hitl_agent",
            input_payload={"foo": "hello "},
            expected_status="interrupted",
            assert_output=_assert_subgraph_hitl,
        )

    print("All sample graphs passed live HTTP end-to-end flow.")


if __name__ == "__main__":
    main()
