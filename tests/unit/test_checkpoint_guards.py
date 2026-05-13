import pytest

from agentseek_api.services.langgraph_service import ensure_sync_checkpoint_mode


def test_raises_when_async_checkpoint_requested() -> None:
    with pytest.raises(RuntimeError, match="sync-oriented"):
        ensure_sync_checkpoint_mode(requested_async=True)
