import asyncio
import json
from datetime import datetime, UTC
from uuid import UUID

import pytest

from agentseek_api.services.sse import iter_with_sse_keepalives, safe_json_dumps


async def _slow_source():
    await asyncio.sleep(10)
    yield "never"


@pytest.mark.asyncio
async def test_keepalive_cleanup_cancels_pending_task() -> None:
    """Covers the finally branch that cancels pending task on early exit."""
    gen = iter_with_sse_keepalives(_slow_source(), interval_seconds=0.01)
    result = await gen.__anext__()
    assert result is None  # keepalive fired
    await gen.aclose()


class _FakeModel:
    def model_dump(self):
        return {"key": "value"}


def test_safe_json_dumps_handles_model_dump() -> None:
    result = json.loads(safe_json_dumps({"m": _FakeModel()}))
    assert result == {"m": {"key": "value"}}


def test_safe_json_dumps_handles_datetime() -> None:
    dt = datetime(2026, 1, 1, tzinfo=UTC)
    result = json.loads(safe_json_dumps({"t": dt}))
    assert result["t"] == dt.isoformat()


def test_safe_json_dumps_handles_uuid() -> None:
    uid = UUID("12345678-1234-5678-1234-567812345678")
    result = json.loads(safe_json_dumps({"id": uid}))
    assert result["id"] == str(uid)


def test_safe_json_dumps_handles_bytes() -> None:
    result = json.loads(safe_json_dumps({"b": b"hello"}))
    assert result["b"] == "hello"


def test_safe_json_dumps_handles_set_and_frozenset() -> None:
    result = json.loads(safe_json_dumps({"s": {1, 2}, "f": frozenset([3])}))
    assert sorted(result["s"]) == [1, 2]
    assert result["f"] == [3]


def test_safe_json_dumps_preserves_non_ascii_characters() -> None:
    result = safe_json_dumps({"text": "\u9644\u4ef64"})
    assert "\u9644\u4ef64" in result
    assert "\\u9644\\u4ef64" not in result


def test_safe_json_dumps_raises_for_unknown_type() -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        safe_json_dumps({"x": object()})
