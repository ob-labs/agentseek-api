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


@pytest.mark.asyncio
async def test_thread_stream_lua_preserves_empty_arrays_and_escapes_event_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _TEST_REDIS_URL is not None
    redis = from_url(_TEST_REDIS_URL, decode_responses=True)
    thread_id = 'thread-"' + "\\" + uuid4().hex[:8]
    sequence_key = f"agentseek:threads:stream-seq:{thread_id}"
    stream_key = f"agentseek:threads:stream:{thread_id}"
    payload = {
        "method": "messages/partial",
        "params": {
            "data": {
                "tool_calls": [],
                "invalid_tool_calls": [],
            }
        },
    }
    monkeypatch.setattr(stream_module, "_redis_client", redis)

    try:
        seq, event = await stream_module.append_redis_thread_stream_event(thread_id, payload)
        rows = await redis.xrange(stream_key, min="-", max="+")
    finally:
        await redis.delete(sequence_key, stream_key)
        await redis.aclose()

    assert seq == 1
    assert event["event_id"] == f"{thread_id}:1"
    assert event["params"]["data"]["tool_calls"] == []
    assert event["params"]["data"]["invalid_tool_calls"] == []
    assert len(rows) == 1
    stored_event = json.loads(rows[0][1]["payload"])
    assert stored_event == event
