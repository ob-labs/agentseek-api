import pytest
from sqlalchemy.orm.exc import StaleDataError

from agentseek_api.services import redis_delivery


@pytest.mark.parametrize("error, expected_attempts", [
    (StaleDataError("stale acknowledgment"), 8),
    (RuntimeError("Redis unavailable"), 1),
])
async def test_delivery_retry_is_bounded_and_specific(monkeypatch, error, expected_attempts):
    attempts = 0

    async def failing_transaction(operation):
        nonlocal attempts
        attempts += 1
        raise error

    monkeypatch.setattr(redis_delivery, "retry_transaction", failing_transaction)
    with pytest.raises(type(error), match=str(error)):
        await redis_delivery._deliver("operation")
    assert attempts == expected_attempts
