import json

import httpx
import pytest


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_real_api_flow(e2e_base_url: str) -> None:
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(base_url=e2e_base_url, timeout=timeout) as client:
        assistant_create = await client.post("/assistants", json={"name": "e2e-assistant", "graph_id": "default"})
        assert assistant_create.status_code == 200
        assistant_id = assistant_create.json()["assistant_id"]

        assistants_list = await client.get("/assistants")
        assert assistants_list.status_code == 200
        assert any(item["assistant_id"] == assistant_id for item in assistants_list.json())

        assistant_get = await client.get(f"/assistants/{assistant_id}")
        assert assistant_get.status_code == 200
        assert assistant_get.json()["assistant_id"] == assistant_id

        thread_create = await client.post("/threads", json={"metadata": {"suite": "e2e"}})
        assert thread_create.status_code == 200
        thread_id = thread_create.json()["thread_id"]

        thread_list = await client.get("/threads")
        assert thread_list.status_code == 200
        assert any(item["thread_id"] == thread_id for item in thread_list.json())

        thread_get = await client.get(f"/threads/{thread_id}")
        assert thread_get.status_code == 200
        assert thread_get.json()["thread_id"] == thread_id

        run_create = await client.post(
            f"/threads/{thread_id}/runs",
            json={"assistant_id": assistant_id, "input": {"message": "hello e2e"}},
        )
        assert run_create.status_code == 200
        run_id = run_create.json()["run_id"]

        run_get = await client.get(f"/threads/{thread_id}/runs/{run_id}")
        assert run_get.status_code == 200
        assert run_get.json()["run_id"] == run_id

        runs_list = await client.get(f"/threads/{thread_id}/runs")
        assert runs_list.status_code == 200
        assert any(item["run_id"] == run_id for item in runs_list.json())

        wait_response = await client.get(f"/threads/{thread_id}/runs/{run_id}/wait")
        assert wait_response.status_code == 200
        assert wait_response.json()["status"] == "success"

        stream_response = await client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
        assert stream_response.status_code == 200
        body_text = stream_response.text
        assert "event: end" in body_text

        stateless_run = await client.post("/runs", json={"assistant_id": assistant_id, "input": {"mode": "stateless"}})
        assert stateless_run.status_code == 200
        assert stateless_run.json()["assistant_id"] == assistant_id


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stream_payload_contains_json_event(e2e_base_url: str) -> None:
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(base_url=e2e_base_url, timeout=timeout) as client:
        assistant_create = await client.post("/assistants", json={"name": "stream-assistant", "graph_id": "default"})
        assert assistant_create.status_code == 200
        assistant_id = assistant_create.json()["assistant_id"]

        thread_create = await client.post("/threads", json={"metadata": {"suite": "stream"}})
        assert thread_create.status_code == 200
        thread_id = thread_create.json()["thread_id"]

        run_create = await client.post(
            f"/threads/{thread_id}/runs",
            json={"assistant_id": assistant_id, "input": {"message": "stream"}},
        )
        assert run_create.status_code == 200
        run_id = run_create.json()["run_id"]

        stream_response = await client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
        assert stream_response.status_code == 200
        lines = [line for line in stream_response.text.splitlines() if line.startswith("data: ")]
        assert lines
        payload = json.loads(lines[-1].replace("data: ", "", 1))
        assert payload["run_id"] == run_id
