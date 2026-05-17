import os

import httpx


def main() -> None:
    base_url = os.getenv("EXAMPLE_BASE_URL", "http://127.0.0.1:2026")
    user_id = os.getenv("EXAMPLE_USER_ID", "example-user")
    headers = {"x-user-id": user_id}

    with httpx.Client(base_url=base_url, timeout=30.0, headers=headers, trust_env=False) as client:
        assistant = client.post("/assistants", json={"name": "live-example-assistant", "graph_id": "default"})
        assistant.raise_for_status()
        assistant_id = assistant.json()["assistant_id"]

        thread = client.post("/threads", json={"metadata": {"source": "live-example"}})
        thread.raise_for_status()
        thread_id = thread.json()["thread_id"]

        run = client.post(
            f"/threads/{thread_id}/runs",
            json={"assistant_id": assistant_id, "input": {"message": "hello-live"}},
        )
        run.raise_for_status()
        run_id = run.json()["run_id"]

        waited = client.get(f"/threads/{thread_id}/runs/{run_id}/wait")
        waited.raise_for_status()
        assert waited.json()["status"] == "success"

        stream = client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
        stream.raise_for_status()
        assert "event: end" in stream.text

    print("Live HTTP end-to-end flow passed")


if __name__ == "__main__":
    main()
