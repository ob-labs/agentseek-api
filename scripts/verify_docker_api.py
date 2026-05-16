from __future__ import annotations

import argparse
import json
from urllib import error as urllib_error
from urllib import request as urllib_request


def _request(
    *,
    base_url: str,
    path: str,
    method: str = "GET",
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
    expected_status: int = 200,
) -> tuple[int, dict[str, object] | list[object] | str | None, str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    req = urllib_request.Request(
        f"{base_url}{path}",
        data=data,
        method=method,
        headers=request_headers,
    )
    try:
        with urllib_request.urlopen(req, timeout=30.0) as response:
            status = response.status
            raw_body = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")
    except urllib_error.HTTPError as exc:
        status = exc.code
        raw_body = exc.read().decode("utf-8")
        content_type = exc.headers.get("Content-Type", "")
    if status != expected_status:
        raise AssertionError(f"{method} {path} returned {status}, expected {expected_status}: {raw_body}")
    if not raw_body:
        return status, None, content_type
    if "application/json" in content_type:
        return status, json.loads(raw_body), content_type
    return status, raw_body, content_type


def _stream_payloads(stream_text: str) -> list[dict[str, object]]:
    return [
        json.loads(line.replace("data: ", "", 1))
        for line in stream_text.splitlines()
        if line.startswith("data: ")
    ]


def _assert_sample_run(
    *,
    base_url: str,
    user_headers: dict[str, str],
    graph_id: str,
    input_payload: dict[str, object],
    expected_status: str = "success",
) -> tuple[dict[str, object], list[dict[str, object]]]:
    _, assistant, _ = _request(
        base_url=base_url,
        path="/assistants",
        method="POST",
        payload={"name": f"docker-sample-{graph_id}", "graph_id": graph_id},
    )
    assert isinstance(assistant, dict)
    assistant_id = str(assistant["assistant_id"])

    _, thread, _ = _request(
        base_url=base_url,
        path="/threads",
        method="POST",
        payload={"metadata": {"suite": "docker-samples", "graph_id": graph_id}},
        headers=user_headers,
    )
    assert isinstance(thread, dict)
    thread_id = str(thread["thread_id"])

    _, run, _ = _request(
        base_url=base_url,
        path=f"/threads/{thread_id}/runs",
        method="POST",
        payload={"assistant_id": assistant_id, "input": input_payload},
        headers=user_headers,
    )
    assert isinstance(run, dict)
    run_id = str(run["run_id"])

    _, waited, _ = _request(
        base_url=base_url,
        path=f"/threads/{thread_id}/runs/{run_id}/wait",
        headers=user_headers,
    )
    assert isinstance(waited, dict)
    assert waited["status"] == expected_status

    _, stream_body, stream_content_type = _request(
        base_url=base_url,
        path=f"/threads/{thread_id}/runs/{run_id}/stream",
        headers=user_headers,
    )
    assert isinstance(stream_body, str)
    assert "text/event-stream" in stream_content_type
    return waited, _stream_payloads(stream_body)


def _assert_common_flow(base_url: str) -> None:
    alice = {"x-user-id": "alice"}
    bob = {"x-user-id": "bob"}

    _, health, _ = _request(base_url=base_url, path="/health")
    assert health == {"status": "healthy"}

    _, info, _ = _request(base_url=base_url, path="/info")
    assert isinstance(info, dict)
    assert info["flags"]["assistants"] is True
    assert info["flags"]["threads"] is True
    assert info["flags"]["runs"] is True
    assert isinstance(info["version"], str) and info["version"]

    _, assistants, _ = _request(base_url=base_url, path="/assistants")
    assert isinstance(assistants, list)

    _, created_assistant, _ = _request(
        base_url=base_url,
        path="/assistants",
        method="POST",
        payload={"name": "docker-external", "graph_id": "external_hello"},
    )
    assert isinstance(created_assistant, dict)
    assistant_id = str(created_assistant["assistant_id"])

    _, fetched_assistant, _ = _request(base_url=base_url, path=f"/assistants/{assistant_id}")
    assert isinstance(fetched_assistant, dict)
    assert fetched_assistant["assistant_id"] == assistant_id

    _, listed_assistants, _ = _request(base_url=base_url, path="/assistants")
    assert isinstance(listed_assistants, list)
    assert any(item["assistant_id"] == assistant_id for item in listed_assistants)

    _, created_thread, _ = _request(
        base_url=base_url,
        path="/threads",
        method="POST",
        payload={"metadata": {"suite": "docker-full"}},
        headers=alice,
    )
    assert isinstance(created_thread, dict)
    thread_id = str(created_thread["thread_id"])
    assert created_thread["user_id"] == "alice"

    _, listed_threads, _ = _request(base_url=base_url, path="/threads", headers=alice)
    assert isinstance(listed_threads, list)
    assert any(item["thread_id"] == thread_id for item in listed_threads)

    _, fetched_thread, _ = _request(base_url=base_url, path=f"/threads/{thread_id}", headers=alice)
    assert isinstance(fetched_thread, dict)
    assert fetched_thread["thread_id"] == thread_id
    assert fetched_thread["user_id"] == "alice"

    _, other_threads, _ = _request(base_url=base_url, path="/threads", headers=bob)
    assert isinstance(other_threads, list)
    assert all(item["thread_id"] != thread_id for item in other_threads)

    _, created_run, _ = _request(
        base_url=base_url,
        path=f"/threads/{thread_id}/runs",
        method="POST",
        payload={"assistant_id": assistant_id, "input": {"message": "hello-from-docker"}},
        headers=alice,
    )
    assert isinstance(created_run, dict)
    run_id = str(created_run["run_id"])

    _, fetched_run, _ = _request(base_url=base_url, path=f"/threads/{thread_id}/runs/{run_id}", headers=alice)
    assert isinstance(fetched_run, dict)
    assert fetched_run["run_id"] == run_id

    _, listed_runs, _ = _request(base_url=base_url, path=f"/threads/{thread_id}/runs", headers=alice)
    assert isinstance(listed_runs, list)
    assert any(item["run_id"] == run_id for item in listed_runs)

    _, waited_run, _ = _request(base_url=base_url, path=f"/threads/{thread_id}/runs/{run_id}/wait", headers=alice)
    assert isinstance(waited_run, dict)
    assert waited_run["status"] == "success", waited_run
    assert waited_run["output"]["final_text"] == "external graph heard: hello-from-docker"

    _, missing_for_bob, _ = _request(
        base_url=base_url,
        path=f"/threads/{thread_id}/runs/{run_id}",
        headers=bob,
        expected_status=404,
    )
    assert isinstance(missing_for_bob, dict)

    _, bob_runs, _ = _request(base_url=base_url, path=f"/threads/{thread_id}/runs", headers=bob)
    assert bob_runs == []

    _, stream_body, stream_content_type = _request(
        base_url=base_url,
        path=f"/threads/{thread_id}/runs/{run_id}/stream",
        headers=alice,
    )
    assert isinstance(stream_body, str)
    assert "text/event-stream" in stream_content_type
    payloads = _stream_payloads(stream_body)
    assert any(payload["event"] == "start" for payload in payloads)
    assert any(payload["event"] == "end" and payload.get("status") == "success" for payload in payloads)

    _, stateless_run, _ = _request(
        base_url=base_url,
        path="/runs",
        method="POST",
        payload={"assistant_id": assistant_id, "input": {"message": "hello-stateless"}},
        headers=alice,
    )
    assert isinstance(stateless_run, dict)
    stateless_thread_id = str(stateless_run["thread_id"])
    stateless_run_id = str(stateless_run["run_id"])
    assert stateless_run["assistant_id"] == assistant_id

    _, stateless_thread, _ = _request(base_url=base_url, path=f"/threads/{stateless_thread_id}", headers=alice)
    assert isinstance(stateless_thread, dict)
    assert stateless_thread["thread_id"] == stateless_thread_id

    _, stateless_run_get, _ = _request(
        base_url=base_url,
        path=f"/threads/{stateless_thread_id}/runs/{stateless_run_id}",
        headers=alice,
    )
    assert isinstance(stateless_run_get, dict)
    assert stateless_run_get["run_id"] == stateless_run_id

    _, hitl_assistant, _ = _request(
        base_url=base_url,
        path="/assistants",
        method="POST",
        payload={"name": "docker-hitl", "graph_id": "subgraph_hitl_agent"},
    )
    assert isinstance(hitl_assistant, dict)
    hitl_assistant_id = str(hitl_assistant["assistant_id"])

    _, hitl_thread, _ = _request(
        base_url=base_url,
        path="/threads",
        method="POST",
        payload={"metadata": {"suite": "docker-resume"}},
        headers=alice,
    )
    assert isinstance(hitl_thread, dict)
    hitl_thread_id = str(hitl_thread["thread_id"])

    _, created_hitl_run, _ = _request(
        base_url=base_url,
        path=f"/threads/{hitl_thread_id}/runs",
        method="POST",
        payload={"assistant_id": hitl_assistant_id, "input": {"foo": "hello "}},
        headers=alice,
    )
    assert isinstance(created_hitl_run, dict)
    interrupted_run_id = str(created_hitl_run["run_id"])

    _, waited_interrupt, _ = _request(
        base_url=base_url,
        path=f"/threads/{hitl_thread_id}/runs/{interrupted_run_id}/wait",
        headers=alice,
    )
    assert isinstance(waited_interrupt, dict)
    assert waited_interrupt["status"] == "interrupted"
    assert waited_interrupt["interrupts"][0]["value"] == "Provide value:"

    _, resumed_run, _ = _request(
        base_url=base_url,
        path=f"/threads/{hitl_thread_id}/runs/{interrupted_run_id}/resume",
        method="POST",
        payload={"resume": "world"},
        headers=alice,
    )
    assert isinstance(resumed_run, dict)
    assert resumed_run["run_id"] == interrupted_run_id

    _, resumed_wait, _ = _request(
        base_url=base_url,
        path=f"/threads/{hitl_thread_id}/runs/{interrupted_run_id}/wait",
        headers=alice,
    )
    assert isinstance(resumed_wait, dict)
    assert resumed_wait["status"] == "success"
    assert resumed_wait["output"]["state"]["foo"].endswith("world")

    _, resumed_stream, _ = _request(
        base_url=base_url,
        path=f"/threads/{hitl_thread_id}/runs/{interrupted_run_id}/stream",
        headers=alice,
    )
    assert isinstance(resumed_stream, str)
    end_statuses = {
        payload["status"]
        for payload in _stream_payloads(resumed_stream)
        if payload["event"] == "end"
    }
    assert "success" in end_statuses

    stress_waited, _ = _assert_sample_run(
        base_url=base_url,
        user_headers=alice,
        graph_id="stress_test",
        input_payload={"delay": 0.0, "steps": 2},
    )
    stress_output = stress_waited["output"]
    assert isinstance(stress_output, dict)
    assert stress_output["final_json"]["steps_completed"] == 2

    subgraph_waited, _ = _assert_sample_run(
        base_url=base_url,
        user_headers=alice,
        graph_id="subgraph_agent",
        input_payload={"delay": 0.0, "steps": 1},
    )
    subgraph_output = subgraph_waited["output"]
    assert isinstance(subgraph_output, dict)
    assert subgraph_output["final_json"]["status"] == "completed"

    react_waited, react_payloads = _assert_sample_run(
        base_url=base_url,
        user_headers=alice,
        graph_id="react_agent",
        input_payload={"message": "what is the meaning of life?"},
    )
    react_output = react_waited["output"]
    assert isinstance(react_output, dict)
    assert "42" in str(react_output["final_text"])
    assert any(payload["event"] == "tool_start" and payload["name"] == "lookup" for payload in react_payloads)

    stress_tool_waited, stress_tool_payloads = _assert_sample_run(
        base_url=base_url,
        user_headers=alice,
        graph_id="stress_tool_agent",
        input_payload={"delay": 0.0, "steps": 3},
    )
    stress_tool_output = stress_tool_waited["output"]
    assert isinstance(stress_tool_output, dict)
    assert stress_tool_output["final_json"]["steps_completed"] == 3
    tool_messages = [message for message in stress_tool_output["transcript"] if message["type"] == "ToolMessage"]
    assert len(tool_messages) == 3
    tool_starts = [
        payload for payload in stress_tool_payloads if payload["event"] == "tool_start" and payload["name"] == "slow_process"
    ]
    assert len(tool_starts) == 3


def _assert_smoke_flow(base_url: str) -> None:
    headers = {"x-user-id": "autobuild"}
    _assert_health(base_url)

    _, created_assistant, _ = _request(
        base_url=base_url,
        path="/assistants",
        method="POST",
        payload={"name": "docker-autobuild", "graph_id": "external_hello"},
    )
    assert isinstance(created_assistant, dict)
    assistant_id = str(created_assistant["assistant_id"])

    _, created_thread, _ = _request(
        base_url=base_url,
        path="/threads",
        method="POST",
        payload={"metadata": {"suite": "docker-autobuild"}},
        headers=headers,
    )
    assert isinstance(created_thread, dict)
    assert created_thread["user_id"] == "autobuild"
    thread_id = str(created_thread["thread_id"])

    _, waited_run, _ = _request(
        base_url=base_url,
        path=f"/threads/{thread_id}/runs",
        method="POST",
        payload={"assistant_id": assistant_id, "input": {"message": "hello-from-autobuild"}},
        headers=headers,
    )
    assert isinstance(waited_run, dict)
    run_id = str(waited_run["run_id"])

    _, final_run, _ = _request(
        base_url=base_url,
        path=f"/threads/{thread_id}/runs/{run_id}/wait",
        headers=headers,
    )
    assert isinstance(final_run, dict)
    assert final_run["status"] == "success"
    assert final_run["output"]["final_text"] == "external graph heard: hello-from-autobuild"


def _assert_health(base_url: str) -> None:
    _, health, _ = _request(base_url=base_url, path="/health")
    assert health == {"status": "healthy"}
    _, info, _ = _request(base_url=base_url, path="/info")
    assert isinstance(info, dict)
    assert isinstance(info["version"], str) and info["version"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    args = parser.parse_args()

    if args.mode == "full":
        _assert_common_flow(args.base_url)
    else:
        _assert_smoke_flow(args.base_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
