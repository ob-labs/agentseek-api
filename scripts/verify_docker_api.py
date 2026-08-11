from __future__ import annotations

import argparse
import http.client
import json
from urllib import error as urllib_error
from urllib import parse as urllib_parse
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


def _sse_frames(stream_text: str) -> list[tuple[int | None, str, dict[str, object]]]:
    """Parse SSE text into (id, event, data) frames. ``id`` may be absent."""
    frames: list[tuple[int | None, str, dict[str, object]]] = []
    current_id: int | None = None
    current_event = ""
    current_data: list[str] = []
    for line in stream_text.splitlines():
        if line.startswith("id: "):
            current_id = int(line[len("id: "):].strip())
        elif line.startswith("event: "):
            current_event = line[len("event: "):].strip()
        elif line.startswith("data: "):
            current_data.append(line[len("data: "):].strip())
        elif line == "" and current_data:
            frames.append((current_id, current_event, json.loads("".join(current_data))))
            current_id, current_event, current_data = None, "", []
    return frames


def _read_sse_prefix_then_disconnect(
    *,
    base_url: str,
    path: str,
    headers: dict[str, str],
    max_frames: int,
    timeout_seconds: float = 15.0,
) -> list[tuple[int | None, str, dict[str, object]]]:
    """Connect to an SSE endpoint, read up to ``max_frames`` frames, then close
    the connection to simulate a client disconnecting mid-run.

    ``urllib`` (used by ``_request``) blocks until the whole body is read, so a
    mid-stream disconnect has to be driven with a lower-level ``http.client``
    connection that we can tear down after a few frames.
    """
    parsed = urllib_parse.urlsplit(base_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout_seconds)
    frames: list[tuple[int | None, str, dict[str, object]]] = []
    try:
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        current_id: int | None = None
        current_event = ""
        current_data: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("id: "):
                current_id = int(line[len("id: "):].strip())
            elif line.startswith("event: "):
                current_event = line[len("event: "):].strip()
            elif line.startswith("data: "):
                current_data.append(line[len("data: "):].strip())
            elif line == "" and current_data:
                frames.append((current_id, current_event, json.loads("".join(current_data))))
                current_id, current_event, current_data = None, "", []
                if len(frames) >= max_frames:
                    break
    finally:
        conn.close()
    return frames



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


def _assert_custom_auth_default_identity(
    *,
    base_url: str,
    other_headers: dict[str, str],
) -> None:
    _, created_thread, _ = _request(
        base_url=base_url,
        path="/threads",
        method="POST",
        payload={"metadata": {"suite": "docker-auth-default"}},
    )
    assert isinstance(created_thread, dict)
    thread_id = str(created_thread["thread_id"])

    _, default_threads, _ = _request(base_url=base_url, path="/threads/search", method="POST", payload={})
    assert isinstance(default_threads, list)
    assert any(item["thread_id"] == thread_id for item in default_threads)

    _, hidden_from_other_user, _ = _request(
        base_url=base_url,
        path=f"/threads/{thread_id}",
        headers=other_headers,
        expected_status=404,
    )
    assert isinstance(hidden_from_other_user, dict)


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

    _, assistants, _ = _request(base_url=base_url, path="/assistants/search", method="POST", payload={})
    assert isinstance(assistants, list)

    _assert_custom_auth_default_identity(base_url=base_url, other_headers=alice)

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

    _, listed_assistants, _ = _request(base_url=base_url, path="/assistants/search", method="POST", payload={})
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

    _, listed_threads, _ = _request(base_url=base_url, path="/threads/search", method="POST", payload={}, headers=alice)
    assert isinstance(listed_threads, list)
    assert any(item["thread_id"] == thread_id for item in listed_threads)

    _, fetched_thread, _ = _request(base_url=base_url, path=f"/threads/{thread_id}", headers=alice)
    assert isinstance(fetched_thread, dict)
    assert fetched_thread["thread_id"] == thread_id

    _, other_threads, _ = _request(base_url=base_url, path="/threads/search", method="POST", payload={}, headers=bob)
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

    _, bob_runs, _ = _request(base_url=base_url, path=f"/threads/{thread_id}/runs", headers=bob, expected_status=404)
    assert isinstance(bob_runs, dict)

    _, stream_body, stream_content_type = _request(
        base_url=base_url,
        path=f"/threads/{thread_id}/runs/{run_id}/stream",
        headers=alice,
    )
    assert isinstance(stream_body, str)
    assert "text/event-stream" in stream_content_type
    payloads = _stream_payloads(stream_body)
    assert any(isinstance(payload, dict) and payload.get("event") == "start" for payload in payloads)
    assert any(
        isinstance(payload, dict)
        and payload.get("event") == "end"
        and payload.get("status") == "success"
        for payload in payloads
    )

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
        if isinstance(payload, dict) and payload.get("event") == "end"
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
    assert any(
        isinstance(payload, dict)
        and payload.get("event") == "tool-started"
        and payload.get("tool_name") == "lookup"
        for payload in react_payloads
    )

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
        payload
        for payload in stress_tool_payloads
        if isinstance(payload, dict)
        and payload.get("event") == "tool-started"
        and payload.get("tool_name") == "slow_process"
    ]
    assert len(tool_starts) == 3

    _assert_run_stream_reconnect_exactly_once(base_url=base_url, user_headers=alice)


def _assert_run_stream_reconnect_exactly_once(*, base_url: str, user_headers: dict[str, str]) -> None:
    """Regression for the run-stream reconnect contract.

    A client that connects to ``GET /runs/{id}/stream`` while the run is still
    executing, disconnects, and reconnects with ``Last-Event-ID`` must not
    replay frames already delivered and must not lose frames produced while it
    was disconnected (exactly-once). This runs against the real HTTP boundary
    and, in the redis-durable job, against the real Redis run-stream store —
    the path the in-process pytest regression covers with a faked store.
    """
    _, assistant, _ = _request(
        base_url=base_url,
        path="/assistants",
        method="POST",
        payload={"name": "docker-reconnect", "graph_id": "stress_tool_agent"},
    )
    assert isinstance(assistant, dict)
    assistant_id = str(assistant["assistant_id"])

    _, thread, _ = _request(
        base_url=base_url,
        path="/threads",
        method="POST",
        payload={"metadata": {"suite": "docker-reconnect"}},
        headers=user_headers,
    )
    assert isinstance(thread, dict)
    thread_id = str(thread["thread_id"])

    # ~4.5s total runtime (3 steps * 1.5s) guarantees a mid-run window.
    _, run, _ = _request(
        base_url=base_url,
        path=f"/threads/{thread_id}/runs",
        method="POST",
        payload={"assistant_id": assistant_id, "input": {"delay": 1.5, "steps": 3}},
        headers=user_headers,
    )
    assert isinstance(run, dict)
    run_id = str(run["run_id"])

    stream_path = f"/threads/{thread_id}/runs/{run_id}/stream"

    # Phase 1: connect mid-run, read a few frames, then disconnect.
    phase1 = _read_sse_prefix_then_disconnect(
        base_url=base_url,
        path=stream_path,
        headers=user_headers,
        max_frames=3,
    )
    if not phase1:
        phase1 = _read_sse_prefix_then_disconnect(
            base_url=base_url,
            path=stream_path,
            headers=user_headers,
            max_frames=3,
        )
    assert phase1, "expected to observe the run stream while the run is still active"
    phase1_ids = [frame_id for frame_id, _, _ in phase1 if frame_id is not None]
    last_id = int(phase1_ids[-1])

    _, waited, _ = _request(
        base_url=base_url,
        path=f"/threads/{thread_id}/runs/{run_id}/wait",
        headers=user_headers,
    )
    assert isinstance(waited, dict)
    assert waited["status"] == "success"

    # Phase 2: reconnect with Last-Event-ID and assert exactly-once.
    _, phase2_body, _ = _request(
        base_url=base_url,
        path=stream_path,
        headers={**user_headers, "Last-Event-ID": str(last_id)},
    )
    assert isinstance(phase2_body, str)
    phase2_ids = [frame_id for frame_id, _, _ in _sse_frames(phase2_body) if frame_id is not None]

    assert all(frame_id > last_id for frame_id in phase2_ids), (
        f"reconnect replayed an already-delivered id: phase1={phase1_ids} phase2={phase2_ids}"
    )
    all_ids = phase1_ids + phase2_ids
    assert len(all_ids) == len(set(all_ids)), f"duplicate ids delivered across reconnect: {all_ids}"

    _, full_body, _ = _request(base_url=base_url, path=stream_path, headers=user_headers)
    assert isinstance(full_body, str)
    full_ids = [frame_id for frame_id, _, _ in _sse_frames(full_body) if frame_id is not None]
    assert [frame_id for frame_id in full_ids if frame_id > last_id] == phase2_ids, (
        f"reconnect content mismatch: full={full_ids} phase2={phase2_ids}"
    )


def _assert_smoke_flow(base_url: str) -> None:
    headers = {"x-user-id": "autobuild"}
    _assert_health(base_url)
    _assert_custom_auth_default_identity(base_url=base_url, other_headers=headers)

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


def _assert_resume_seed(base_url: str) -> dict[str, str]:
    alice = {"x-user-id": "alice"}
    _, assistant, _ = _request(
        base_url=base_url,
        path="/assistants",
        method="POST",
        payload={"name": "docker-hitl-restart", "graph_id": "subgraph_hitl_agent"},
    )
    assert isinstance(assistant, dict)
    assistant_id = str(assistant["assistant_id"])

    _, thread, _ = _request(
        base_url=base_url,
        path="/threads",
        method="POST",
        payload={"metadata": {"suite": "docker-resume-restart"}},
        headers=alice,
    )
    assert isinstance(thread, dict)
    thread_id = str(thread["thread_id"])

    _, run, _ = _request(
        base_url=base_url,
        path=f"/threads/{thread_id}/runs",
        method="POST",
        payload={"assistant_id": assistant_id, "input": {"foo": "hello "}},
        headers=alice,
    )
    assert isinstance(run, dict)
    run_id = str(run["run_id"])

    _, waited, _ = _request(
        base_url=base_url,
        path=f"/threads/{thread_id}/runs/{run_id}/wait",
        headers=alice,
    )
    assert isinstance(waited, dict)
    assert waited["status"] == "interrupted"

    return {"thread_id": thread_id, "run_id": run_id}


def _assert_resume_check(base_url: str, *, thread_id: str, run_id: str, resume: str) -> None:
    alice = {"x-user-id": "alice"}
    _, resumed, _ = _request(
        base_url=base_url,
        path=f"/threads/{thread_id}/runs/{run_id}/resume",
        method="POST",
        payload={"resume": resume},
        headers=alice,
    )
    assert isinstance(resumed, dict)
    assert resumed["run_id"] == run_id

    _, waited, _ = _request(
        base_url=base_url,
        path=f"/threads/{thread_id}/runs/{run_id}/wait",
        headers=alice,
    )
    assert isinstance(waited, dict)
    assert waited["status"] == "success"
    assert waited["output"]["state"]["foo"].endswith(resume)

    _, run_stream, _ = _request(
        base_url=base_url,
        path=f"/threads/{thread_id}/runs/{run_id}/stream",
        headers=alice,
    )
    assert isinstance(run_stream, str)
    end_statuses = [
        payload["status"]
        for payload in _stream_payloads(run_stream)
        if isinstance(payload, dict) and payload.get("event") == "end"
    ]
    assert "interrupted" in end_statuses
    assert "success" in end_statuses



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--mode", choices=("full", "smoke", "resume-seed", "resume-check"), default="full")
    parser.add_argument("--state-file")
    parser.add_argument("--resume-value", default="world")
    args = parser.parse_args()

    if args.mode == "full":
        _assert_common_flow(args.base_url)
    elif args.mode == "smoke":
        _assert_smoke_flow(args.base_url)
    elif args.mode == "resume-seed":
        print(json.dumps(_assert_resume_seed(args.base_url)))
    else:
        if not args.state_file:
            raise SystemExit("--state-file is required for --mode resume-check")
        state = json.loads(open(args.state_file, encoding="utf-8").read())
        if not isinstance(state, dict):
            raise SystemExit("State file must contain a JSON object.")
        _assert_resume_check(
            args.base_url,
            thread_id=str(state["thread_id"]),
            run_id=str(state["run_id"]),
            resume=args.resume_value,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
