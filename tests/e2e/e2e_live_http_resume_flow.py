import os

import httpx


def main() -> None:
    base_url = os.getenv("EXAMPLE_BASE_URL", "http://127.0.0.1:2026")
    user_id = os.getenv("EXAMPLE_USER_ID", "resume-user")
    headers = {"x-user-id": user_id}

    with httpx.Client(base_url=base_url, timeout=30.0, headers=headers) as client:
        assistant = client.post("/assistants", json={"name": "live-resume-assistant", "graph_id": "subgraph_hitl_agent"})
        assistant.raise_for_status()
        assistant_id = assistant.json()["assistant_id"]

        thread = client.post("/threads", json={"metadata": {"source": "live-resume"}})
        thread.raise_for_status()
        thread_id = thread.json()["thread_id"]

        run = client.post(
            f"/threads/{thread_id}/runs",
            json={"assistant_id": assistant_id, "input": {"foo": "hello "}},
        )
        run.raise_for_status()
        run_body = run.json()
        run_id = run_body["run_id"]

        waited = client.get(f"/threads/{thread_id}/runs/{run_id}/wait")
        waited.raise_for_status()
        waited_body = waited.json()
        assert waited_body["status"] == "interrupted"
        assert waited_body["interrupts"][0]["value"] == "Provide value:"

        resumed = client.post(f"/threads/{thread_id}/runs/{run_id}/resume", json={"resume": "world"})
        resumed.raise_for_status()
        resumed_body = resumed.json()

        resumed_wait = client.get(f"/threads/{thread_id}/runs/{run_id}/wait")
        resumed_wait.raise_for_status()
        resumed_wait_body = resumed_wait.json()
        assert resumed_wait_body["status"] == "success"
        assert resumed_wait_body["output"]["state"]["foo"].endswith("world")
        assert resumed_body["run_id"] == run_id

        stream = client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
        stream.raise_for_status()
        assert '"status": "success"' in stream.text

    print("Live HTTP resume flow passed")


if __name__ == "__main__":
    main()
