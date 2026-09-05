import json
import os
from uuid import uuid4

import pytest
from redis.asyncio import from_url

from agentseek_api.services import stream_persistence as stream_module


_TEST_REDIS_URL = os.getenv("AGENTSEEK_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(
    not _TEST_REDIS_URL,
    reason="AGENTSEEK_TEST_REDIS_URL is required for live Redis script tests",
)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {
                "method": "messages/partial",
                "params": {
                    "data": {
                        "tool_calls": [],
                        "invalid_tool_calls": [],
                    }
                },
            },
            id="message-with-empty-arrays",
        ),
        pytest.param({}, id="empty-object"),
        pytest.param(
            {
                "type": "payload-event",
                "event_id": "payload-event-id",
                "seq": 0,
                "method": "values",
            },
            id="reserved-envelope-fields",
        ),
    ],
)
@pytest.mark.asyncio
async def test_thread_stream_lua_splices_header_without_reencoding_payload(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    assert _TEST_REDIS_URL is not None
    redis = from_url(_TEST_REDIS_URL, decode_responses=True)
    thread_id = 'thread-"' + "\\" + uuid4().hex[:8]
    sequence_key = f"agentseek:threads:stream-seq:{thread_id}"
    stream_key = f"agentseek:threads:stream:{thread_id}"
    monkeypatch.setattr(stream_module, "_redis_client", redis)

    try:
        seq, event = await stream_module.append_redis_thread_stream_event(thread_id, payload)
        rows = await redis.xrange(stream_key, min="-", max="+")
    finally:
        await redis.delete(sequence_key, stream_key)
        await redis.aclose()

    assert seq == 1
    assert event["type"] == "event"
    assert event["event_id"] == f"{thread_id}:1"
    assert event["seq"] == 1
    payload_body = {key: value for key, value in payload.items() if key not in {"type", "event_id", "seq"}}
    expected_event = {"type": "event", "event_id": f"{thread_id}:1", "seq": 1, **payload_body}
    assert event == expected_event
    assert len(rows) == 1
    stored_payload = rows[0][1]["payload"]
    stored_event = json.loads(stored_payload)
    assert stored_event == event
    assert stored_payload == json.dumps(expected_event, ensure_ascii=False, separators=(",", ":"))
