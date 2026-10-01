from fastapi.testclient import TestClient


def test_ordering_setup_waits_for_background_runs(client: TestClient, monkeypatch) -> None:
    import asyncio
    from agentseek_api.services.run_jobs import execute_run_job

    class DelayedExecutor:
        async def submit(self, job):
            async def execute():
                await asyncio.sleep(0.05)
                await execute_run_job(job)
            asyncio.create_task(execute())

    monkeypatch.setattr("agentseek_api.services.run_preparation.get_executor", DelayedExecutor)
    thread_id, _, run_ids = _setup_thread_with_runs(client, count=2)
    assert len(run_ids) == 2
    assert all(client.get(f"/threads/{thread_id}/runs/{run_id}").json()["status"] == "success"
               for run_id in run_ids)


def _setup_thread_with_runs(client: TestClient, count: int = 3) -> tuple[str, str, list[str]]:
    assistant = client.post("/assistants", json={"name": "order-assistant", "graph_id": "default"})
    assert assistant.status_code == 200
    assistant_id = assistant.json()["assistant_id"]

    thread = client.post("/threads", json={"metadata": {"order": True}})
    assert thread.status_code == 200
    thread_id = thread.json()["thread_id"]

    run_ids = []
    for i in range(count):
        resp = client.post(
            f"/threads/{thread_id}/runs",
            json={"assistant_id": assistant_id, "input": {"index": i}},
        )
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]
        # POST returns an accepted run, not a guarantee that it has finished.
        # Recovery may claim it before even the synchronous test executor does.
        completed = client.get(f"/threads/{thread_id}/runs/{run_id}/wait")
        assert completed.status_code == 200
        assert completed.json()["status"] == "success"
        run_ids.append(run_id)
    return thread_id, assistant_id, run_ids


def test_list_runs_latest_first(client: TestClient) -> None:
    thread_id, _, run_ids = _setup_thread_with_runs(client, count=2)

    listed = client.get(f"/threads/{thread_id}/runs")
    assert listed.status_code == 200
    body = listed.json()
    assert len(body) == 2
    assert body[0]["run_id"] == run_ids[1]
    assert body[1]["run_id"] == run_ids[0]


def test_list_runs_limit_and_offset(client: TestClient) -> None:
    thread_id, _, run_ids = _setup_thread_with_runs(client, count=3)

    resp = client.get(f"/threads/{thread_id}/runs", params={"limit": 2})
    assert resp.status_code == 200
    assert len(resp.json()) == 2

    resp = client.get(f"/threads/{thread_id}/runs", params={"limit": 2, "offset": 2})
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["run_id"] == run_ids[0]


def test_list_runs_status_filter(client: TestClient) -> None:
    thread_id, _, run_ids = _setup_thread_with_runs(client, count=2)

    resp = client.get(f"/threads/{thread_id}/runs", params={"status": "success"})
    assert resp.status_code == 200
    for run in resp.json():
        assert run["status"] == "success"

    resp = client.get(f"/threads/{thread_id}/runs", params={"status": "running"})
    assert resp.status_code == 200
    assert resp.json() == []

    resp = client.get(f"/threads/{thread_id}/runs", params={"status": "bogus"})
    assert resp.status_code == 422


def test_list_runs_select_fields(client: TestClient) -> None:
    thread_id, _, _ = _setup_thread_with_runs(client, count=1)

    resp = client.get(f"/threads/{thread_id}/runs", params={"select": ["run_id", "status"]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert set(body[0].keys()) == {"run_id", "status"}
