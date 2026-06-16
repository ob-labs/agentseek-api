import asyncio

from sqlalchemy import select
from fastapi.testclient import TestClient

from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import CronJob


def _create_assistant(client: TestClient) -> str:
    response = client.post("/assistants", json={"name": "cron-assistant", "graph_id": "default"})
    assert response.status_code == 200
    return response.json()["assistant_id"]


def _create_thread(client: TestClient, user_id: str = "default_user") -> str:
    response = client.post("/threads", json={"metadata": {"scope": "cron"}}, headers={"x-user-id": user_id})
    assert response.status_code == 200
    return response.json()["thread_id"]


async def _fetch_cron(cron_id: str) -> CronJob | None:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        return await session.scalar(select(CronJob).where(CronJob.cron_id == cron_id))


def test_create_stateless_cron_persists_and_returns_resource(client: TestClient) -> None:
    assistant_id = _create_assistant(client)

    response = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "timezone": "Asia/Shanghai",
            "input": ["stateless-cron", {"kind": "list-payload"}],
            "metadata": {"source": "integration"},
            "config": {"model": "gpt-test"},
            "context": {"tenant": "acme"},
            "webhook": "https://example.com/hook",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_id"] == assistant_id
    assert body.get("thread_id") is None  # excluded when None via response_model_exclude_none
    assert body["enabled"] is True
    assert body["schedule"] == "FREQ=MINUTELY;INTERVAL=5"
    assert body["timezone"] == "Asia/Shanghai"
    assert body["webhook"] == "https://example.com/hook"
    assert body.get("last_run_at") is None  # excluded when None via response_model_exclude_none
    assert body.get("last_tick_status") is None
    assert body.get("last_error") is None
    assert body["next_run_at"] is not None
    # Spec field next_run_date is present in the response body and aliases next_run_at.
    assert body["next_run_date"] == body["next_run_at"]
    assert body["created_at"] is not None
    assert body["updated_at"] is not None
    assert body["cron_id"]

    persisted = asyncio.run(_fetch_cron(body["cron_id"]))
    assert persisted is not None
    assert persisted.assistant_id == assistant_id
    assert persisted.thread_id is None
    assert persisted.timezone == "Asia/Shanghai"
    assert persisted.schedule == "FREQ=MINUTELY;INTERVAL=5"
    assert persisted.input_json == ["stateless-cron", {"kind": "list-payload"}]
    assert persisted.metadata_json == {"source": "integration"}
    assert persisted.kwargs_json == {"config": {"model": "gpt-test"}, "context": {"tenant": "acme"}, "stream_modes": ["values"]}
    assert persisted.webhook == "https://example.com/hook"
    assert persisted.next_run_at is not None
    assert persisted.next_run_at.isoformat() == body["next_run_at"]


def test_create_thread_cron_persists_thread_and_user_binding(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client, user_id="owner")

    response = client.post(
        f"/threads/{thread_id}/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=HOURLY;INTERVAL=1",
            "input": {"kind": "thread-cron"},
        },
        headers={"x-user-id": "owner"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_id"] == assistant_id
    assert body["thread_id"] == thread_id

    persisted = asyncio.run(_fetch_cron(body["cron_id"]))
    assert persisted is not None
    assert persisted.thread_id == thread_id
    assert persisted.user_id == "owner"
    assert persisted.assistant_id == assistant_id


def test_create_thread_cron_missing_thread_returns_not_found(client: TestClient) -> None:
    assistant_id = _create_assistant(client)

    response = client.post(
        "/threads/does-not-exist/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=DAILY;INTERVAL=1",
            "input": {"kind": "thread-cron"},
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Thread not found"}


def test_search_count_get_patch_and_delete_crons(client: TestClient) -> None:
    assistant_id = _create_assistant(client)

    first = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": {"kind": "search-match"},
            "enabled": True,
        },
    )
    assert first.status_code == 200
    first_body = first.json()

    second = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=HOURLY;INTERVAL=1",
            "input": {"kind": "search-disabled"},
            "enabled": False,
        },
    )
    assert second.status_code == 200

    search_response = client.post(
        "/runs/crons/search",
        json={"assistant_id": assistant_id, "enabled": True, "limit": 10, "offset": 0},
    )
    assert search_response.status_code == 200
    search_body = search_response.json()
    assert [item["cron_id"] for item in search_body["items"]] == [first_body["cron_id"]]

    count_response = client.post(
        "/runs/crons/count",
        json={"assistant_id": assistant_id, "enabled": True},
    )
    assert count_response.status_code == 200
    assert count_response.json() == {"count": 1}

    get_response = client.get(f"/runs/crons/{first_body['cron_id']}")
    assert get_response.status_code == 200
    assert get_response.json()["cron_id"] == first_body["cron_id"]

    patch_response = client.patch(
        f"/runs/crons/{first_body['cron_id']}",
        json={"schedule": "FREQ=MINUTELY;INTERVAL=1", "enabled": False, "input": {"kind": "patched"}},
    )
    assert patch_response.status_code == 200
    patch_body = patch_response.json()
    assert patch_body["schedule"] == "FREQ=MINUTELY;INTERVAL=1"
    assert patch_body["enabled"] is False

    delete_response = client.delete(f"/runs/crons/{first_body['cron_id']}")
    assert delete_response.status_code == 204

    missing_response = client.get(f"/runs/crons/{first_body['cron_id']}")
    assert missing_response.status_code == 404


def test_patch_cron_rejects_explicit_null_input(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": {"kind": "original"},
            "enabled": True,
        },
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    response = client.patch(f"/runs/crons/{cron_id}", json={"input": None})

    assert response.status_code == 400
    assert response.json() == {"detail": "input cannot be null"}


def test_create_cron_rejects_non_http_webhook(client: TestClient) -> None:
    assistant_id = _create_assistant(client)

    response = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": {"kind": "invalid-webhook"},
            "webhook": "/relative/path",
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "webhook must be an absolute http or https URL"}


def test_search_crons_rejects_negative_limit_and_offset(client: TestClient) -> None:
    assistant_id = _create_assistant(client)

    response = client.post(
        "/runs/crons/search",
        json={"assistant_id": assistant_id, "enabled": True, "limit": -1, "offset": -1},
    )

    assert response.status_code == 422


def test_create_stateless_cron_persists_run_control_and_lifecycle(client: TestClient) -> None:
    assistant_id = _create_assistant(client)

    response = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": {"kind": "run-control"},
            "config": {"model": "gpt-test"},
            "context": {"tenant": "acme"},
            "end_time": "2030-01-01T00:00:00+00:00",
            "on_run_completed": "keep",
            "interrupt_before": ["node_a"],
            "interrupt_after": "*",
            "stream_mode": ["values", "messages", "values"],
            "stream_subgraphs": True,
            "stream_resumable": True,
            "durability": "sync",
        },
    )

    assert response.status_code == 200
    body = response.json()
    # CronRead serializes end_time; SQLite drops tzinfo on round-trip so the body
    # mirrors the naive round-tripped value (matches the existing-resource test).
    assert body["end_time"] == "2030-01-01T00:00:00"
    # on_run_completed is a persisted column, not a CronRead response field (per spec),
    # so it is not present in the response body.
    assert "on_run_completed" not in body

    persisted = asyncio.run(_fetch_cron(body["cron_id"]))
    assert persisted is not None
    assert persisted.end_time is not None
    assert persisted.end_time.isoformat() == "2030-01-01T00:00:00"
    assert persisted.on_run_completed == "keep"
    assert persisted.kwargs_json == {
        "config": {"model": "gpt-test"},
        "context": {"tenant": "acme"},
        "stream_modes": ["values", "messages"],
        "interrupt_before": ["node_a"],
        "interrupt_after": "*",
        "durability": "sync",
        "stream_subgraphs": True,
        "stream_resumable": True,
    }


def test_create_stateless_cron_omits_default_run_control_from_kwargs(client: TestClient) -> None:
    assistant_id = _create_assistant(client)

    response = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": {"kind": "defaults"},
            "config": {"model": "gpt-test"},
            "context": {"tenant": "acme"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    # end_time is None -> dropped by response_model_exclude_none on the route.
    assert "end_time" not in body
    # on_run_completed is a persisted column, not a CronRead response field (per spec).
    assert "on_run_completed" not in body

    persisted = asyncio.run(_fetch_cron(body["cron_id"]))
    assert persisted is not None
    assert persisted.end_time is None
    assert persisted.on_run_completed == "delete"
    assert persisted.kwargs_json == {
        "config": {"model": "gpt-test"},
        "context": {"tenant": "acme"},
        "stream_modes": ["values"],
    }


def test_create_thread_cron_persists_multitask_strategy(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client, user_id="owner")

    response = client.post(
        f"/threads/{thread_id}/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=HOURLY;INTERVAL=1",
            "input": {"kind": "thread-cron"},
            "config": {"model": "gpt-test"},
            "context": {"tenant": "acme"},
            "multitask_strategy": "rollback",
        },
        headers={"x-user-id": "owner"},
    )

    assert response.status_code == 200
    body = response.json()

    persisted = asyncio.run(_fetch_cron(body["cron_id"]))
    assert persisted is not None
    assert persisted.kwargs_json["multitask_strategy"] == "rollback"


def test_create_thread_cron_ignores_on_run_completed(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client, user_id="owner")

    # extra="allow" means on_run_completed is silently accepted (not 422) on the
    # thread-cron schema, which has no such field. It must NOT be persisted as a
    # column and must NOT leak into kwargs.
    response = client.post(
        f"/threads/{thread_id}/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=HOURLY;INTERVAL=1",
            "input": {"kind": "thread-cron"},
            "on_run_completed": "keep",
        },
        headers={"x-user-id": "owner"},
    )

    assert response.status_code == 200
    body = response.json()
    # on_run_completed is a persisted column, not a CronRead response field (per spec),
    # so it is never present in the response body.
    assert "on_run_completed" not in body

    persisted = asyncio.run(_fetch_cron(body["cron_id"]))
    assert persisted is not None
    assert persisted.on_run_completed == "delete"
    assert "on_run_completed" not in persisted.kwargs_json


def test_patch_cron_updates_lifecycle_and_run_control(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": {"kind": "original"},
            "config": {"model": "gpt-test"},
            "context": {"tenant": "acme"},
        },
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    response = client.patch(
        f"/runs/crons/{cron_id}",
        json={
            "end_time": "2031-06-01T12:00:00+00:00",
            "durability": "sync",
            "stream_mode": ["updates", "values"],
            "on_run_completed": "keep",
        },
    )

    assert response.status_code == 200
    body = response.json()
    # CronRead serializes end_time; SQLite drops tzinfo on round-trip so the body
    # mirrors the naive round-tripped value (matches the existing-resource tests).
    assert body["end_time"] == "2031-06-01T12:00:00"
    # on_run_completed is a persisted column, not a CronRead response field (per spec),
    # so it is not present in the response body.
    assert "on_run_completed" not in body

    persisted = asyncio.run(_fetch_cron(cron_id))
    assert persisted is not None
    assert persisted.end_time is not None
    assert persisted.end_time.isoformat() == "2031-06-01T12:00:00"
    assert persisted.on_run_completed == "keep"
    assert persisted.kwargs_json["config"] == {"model": "gpt-test"}
    assert persisted.kwargs_json["context"] == {"tenant": "acme"}
    assert persisted.kwargs_json["durability"] == "sync"
    assert persisted.kwargs_json["stream_modes"] == ["updates", "values"]


def test_patch_cron_config_only_preserves_run_control(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": {"kind": "original"},
            "config": {"model": "gpt-test"},
            "context": {"tenant": "acme"},
            "stream_mode": ["messages"],
            "durability": "sync",
        },
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    response = client.patch(
        f"/runs/crons/{cron_id}",
        json={"config": {"model": "gpt-next"}},
    )

    assert response.status_code == 200
    persisted = asyncio.run(_fetch_cron(cron_id))
    assert persisted is not None
    assert persisted.kwargs_json["config"] == {"model": "gpt-next"}
    assert persisted.kwargs_json["context"] == {"tenant": "acme"}
    assert persisted.kwargs_json["stream_modes"] == ["messages"]
    assert persisted.kwargs_json["durability"] == "sync"


def test_patch_thread_cron_updates_multitask_strategy(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    thread_id = _create_thread(client, user_id="owner")
    created = client.post(
        f"/threads/{thread_id}/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=HOURLY;INTERVAL=1",
            "input": {"kind": "thread-cron"},
            "multitask_strategy": "interrupt",
        },
        headers={"x-user-id": "owner"},
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    response = client.patch(f"/runs/crons/{cron_id}", json={"multitask_strategy": "rollback"})
    assert response.status_code == 200

    persisted = asyncio.run(_fetch_cron(cron_id))
    assert persisted is not None
    assert persisted.kwargs_json["multitask_strategy"] == "rollback"


def test_patch_cron_explicit_null_durability_resets_to_default(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": {"kind": "original"},
            "durability": "sync",
        },
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    response = client.patch(f"/runs/crons/{cron_id}", json={"durability": None})
    assert response.status_code == 200

    persisted = asyncio.run(_fetch_cron(cron_id))
    assert persisted is not None
    # Default durability is omitted from kwargs (no junk None stored).
    assert "durability" not in persisted.kwargs_json


def test_patch_cron_explicit_null_clears_interrupt_and_stream_mode(client: TestClient) -> None:
    # The null-clear path must work for every run-control field, not just
    # durability. stream_mode is the subtle one: it is stored under the key
    # "stream_modes" but cleared via the "stream_mode" kwarg (key remap).
    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": {"kind": "original"},
            "interrupt_before": ["node_a"],
            "stream_mode": ["messages", "updates"],
        },
    )
    assert created.status_code == 200
    cron_id = created.json()["cron_id"]

    persisted = asyncio.run(_fetch_cron(cron_id))
    assert persisted.kwargs_json["interrupt_before"] == ["node_a"]
    assert persisted.kwargs_json["stream_modes"] == ["messages", "updates"]

    response = client.patch(
        f"/runs/crons/{cron_id}",
        json={"interrupt_before": None, "stream_mode": None},
    )
    assert response.status_code == 200

    persisted = asyncio.run(_fetch_cron(cron_id))
    assert persisted is not None
    # interrupt_before fully cleared; stream_mode reset to its default.
    assert "interrupt_before" not in persisted.kwargs_json
    assert persisted.kwargs_json["stream_modes"] == ["values"]


def test_create_cron_rejects_unsupported_stream_mode(client: TestClient) -> None:
    # An unsupported stream mode is rejected at creation time (Pydantic's
    # RunStreamMode Literal catches it as 422), not silently persisted to be
    # discovered later when the tick fires.
    assistant_id = _create_assistant(client)
    response = client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": {"kind": "bad-stream"},
            "stream_mode": ["valeus"],
        },
    )
    assert response.status_code == 422


def test_search_crons_sorts_by_next_run_date_asc(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    hourly = client.post(
        "/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=HOURLY;INTERVAL=1", "input": {"k": "hourly"}},
    ).json()
    minutely = client.post(
        "/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=MINUTELY;INTERVAL=1", "input": {"k": "minutely"}},
    ).json()

    response = client.post(
        "/runs/crons/search",
        json={"assistant_id": assistant_id, "sort_by": "next_run_date", "sort_order": "asc"},
    )
    assert response.status_code == 200
    ids = [item["cron_id"] for item in response.json()["items"]]
    assert ids == [minutely["cron_id"], hourly["cron_id"]]


def test_search_crons_filters_by_metadata(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    matching = client.post(
        "/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=MINUTELY;INTERVAL=5", "input": {"k": 1}, "metadata": {"team": "alpha"}},
    ).json()
    client.post(
        "/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=MINUTELY;INTERVAL=5", "input": {"k": 2}, "metadata": {"team": "beta"}},
    )

    response = client.post(
        "/runs/crons/search",
        json={"assistant_id": assistant_id, "metadata": {"team": "alpha"}},
    )
    assert response.status_code == 200
    ids = [item["cron_id"] for item in response.json()["items"]]
    assert ids == [matching["cron_id"]]

    count_response = client.post(
        "/runs/crons/count",
        json={"assistant_id": assistant_id, "metadata": {"team": "alpha"}},
    )
    assert count_response.status_code == 200
    assert count_response.json() == {"count": 1}


def test_search_crons_metadata_filter_known_limitation_non_string_values(client: TestClient) -> None:
    # KNOWN LIMITATION (shared with the thread-search filter): the metadata filter
    # compares metadata_json[key].as_string() == str(value). For a JSON numeric
    # value, as_string() yields the number (not its decimal text), so it never
    # equals str(value) and the row does NOT match. This test PINS that current
    # behavior so any future move to type-aware matching is a deliberate change,
    # not an accident. String metadata values match (see the test above).
    assistant_id = _create_assistant(client)
    client.post(
        "/runs/crons",
        json={
            "assistant_id": assistant_id,
            "schedule": "FREQ=MINUTELY;INTERVAL=5",
            "input": {"k": 1},
            "metadata": {"priority": 7},
        },
    )

    # Numeric filter currently matches nothing (the limitation).
    numeric = client.post(
        "/runs/crons/search",
        json={"assistant_id": assistant_id, "metadata": {"priority": 7}},
    )
    assert numeric.status_code == 200
    assert numeric.json()["items"] == []

    # The cron is still findable by a string-valued filter / no filter.
    unfiltered = client.post("/runs/crons/search", json={"assistant_id": assistant_id})
    assert unfiltered.status_code == 200
    assert len(unfiltered.json()["items"]) == 1


def test_search_crons_select_returns_subset(client: TestClient) -> None:
    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=MINUTELY;INTERVAL=5", "input": {"k": 1}},
    ).json()

    response = client.post(
        "/runs/crons/search",
        json={"assistant_id": assistant_id, "select": ["cron_id", "schedule"]},
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert items == [{"cron_id": created["cron_id"], "schedule": "FREQ=MINUTELY;INTERVAL=5"}]


def test_search_crons_select_omits_null_fields(client: TestClient) -> None:
    # Documents the chosen behavior: a selected-but-null field (e.g. webhook on a
    # cron created without one) is omitted, consistent with the default path's
    # response_model_exclude_none, rather than returned as an explicit null.
    assistant_id = _create_assistant(client)
    created = client.post(
        "/runs/crons",
        json={"assistant_id": assistant_id, "schedule": "FREQ=MINUTELY;INTERVAL=5", "input": {"k": 1}},
    ).json()

    response = client.post(
        "/runs/crons/search",
        json={"assistant_id": assistant_id, "select": ["cron_id", "webhook"]},
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert items == [{"cron_id": created["cron_id"]}]


def test_search_crons_limit_bounds(client: TestClient) -> None:
    assistant_id = _create_assistant(client)

    zero = client.post("/runs/crons/search", json={"assistant_id": assistant_id, "limit": 0})
    assert zero.status_code == 422

    one = client.post("/runs/crons/search", json={"assistant_id": assistant_id, "limit": 1})
    assert one.status_code == 200
