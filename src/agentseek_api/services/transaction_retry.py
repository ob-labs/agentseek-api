"""Whole-transaction retry for known metadata database contention."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from agentseek_api.core.database import db_manager

T = TypeVar("T")


def is_retryable_transaction_error(error: BaseException) -> bool:
    if not isinstance(error, DBAPIError):
        return False
    original = error.orig
    code = getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)
    if code in {"40001", "40P01", "55P03"}:
        return True
    sqlite_code = getattr(original, "sqlite_errorcode", 0)
    if sqlite_code and sqlite_code & 255 in {5, 6}:
        return True
    args = getattr(original, "args", ())
    if args and args[0] in {1205, 1213}:
        return True
    message = str(original).lower()
    return "database is locked" in message or "database table is locked" in message


async def retry_transaction(operation: Callable[[AsyncSession], Awaitable[T]]) -> T:
    """Rollback/close before retrying; never reuse a failed session snapshot."""
    factory = db_manager.get_session_factory()
    for attempt in range(8):
        try:
            async with factory() as session:
                result = await operation(session)
                await session.commit()
                return result
        except DBAPIError as exc:
            if attempt == 7 or not is_retryable_transaction_error(exc):
                raise
        await asyncio.sleep(min(0.005 * 2 ** attempt, 0.2))
    raise AssertionError("unreachable")
