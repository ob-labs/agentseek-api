import asyncio

import pytest

from agentseek_api.services.sse import iter_with_sse_keepalives


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
