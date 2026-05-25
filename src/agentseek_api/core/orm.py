from collections.abc import AsyncIterator
from typing import Any
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Assistant(Base):
    __tablename__ = "assistants"
    assistant_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    graph_id: Mapped[str] = mapped_column(String(255), nullable=False, default="default")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    config_json: Mapped[dict] = mapped_column("config", JSON, default=dict, nullable=False)
    context_json: Mapped[dict] = mapped_column("context", JSON, default=dict, nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    version: Mapped[int] = mapped_column(nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False)


class Thread(Base):
    __tablename__ = "threads"
    thread_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    config_json: Mapped[dict] = mapped_column("config", JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="idle")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False)
    state_updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)


class Run(Base):
    __tablename__ = "runs"
    run_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    thread_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    assistant_id: Mapped[str] = mapped_column(String(36), nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    input_json: Mapped[Any] = mapped_column("input", JSON, default=dict, nullable=False)
    output_json: Mapped[dict | None] = mapped_column("output", JSON, nullable=True)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    kwargs_json: Mapped[dict] = mapped_column("kwargs", JSON, default=dict, nullable=False)
    multitask_strategy: Mapped[str] = mapped_column(String(32), nullable=False, default="enqueue")
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False)


class CronJob(Base):
    __tablename__ = "cron_jobs"

    cron_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    assistant_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    thread_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    schedule: Mapped[str] = mapped_column(String(255), nullable=False)
    timezone: Mapped[str] = mapped_column(String(128), nullable=False, default="UTC")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    input_json: Mapped[dict] = mapped_column("input", JSON, default=dict, nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    kwargs_json: Mapped[dict] = mapped_column("kwargs", JSON, default=dict, nullable=False)
    webhook: Mapped[str | None] = mapped_column(Text, nullable=True)
    max_webhook_attempts: Mapped[int] = mapped_column(nullable=False, default=3)
    next_run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_tick_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False)


class CronTick(Base):
    __tablename__ = "cron_ticks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cron_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    thread_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    scheduler_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    scheduled_for: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="started")
    skip_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    webhook_delivery_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    webhook_attempt_count: Mapped[int] = mapped_column(nullable=False, default=0)
    webhook_last_status_code: Mapped[int | None] = mapped_column(nullable=True)
    webhook_last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    webhook_delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False)


class CronWebhookAttempt(Base):
    __tablename__ = "cron_webhook_attempts"
    __table_args__ = (UniqueConstraint("tick_id", "attempt_number", name="uq_cron_webhook_attempt_tick_attempt"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tick_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status_code: Mapped[int | None] = mapped_column(nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)


class StoreItem(Base):
    __tablename__ = "store_items"
    __table_args__ = (UniqueConstraint("identity_hash", name="uq_store_items_identity_hash"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    identity_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    namespace_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    namespace_json: Mapped[list] = mapped_column("namespace", JSON, default=list, nullable=False)
    key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    value_json: Mapped[dict] = mapped_column("value", JSON, default=dict, nullable=False)
    embedding_json: Mapped[list | None] = mapped_column("embedding", JSON, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False)


class RunStreamEvent(Base):
    __tablename__ = "run_stream_events"
    __table_args__ = (UniqueConstraint("run_id", "seq", name="uq_run_stream_events_run_seq"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    event: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[dict] = mapped_column("payload", JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)


class ThreadStreamEvent(Base):
    __tablename__ = "thread_stream_events"
    __table_args__ = (UniqueConstraint("thread_id", "seq", name="uq_thread_stream_events_thread_seq"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    method: Mapped[str] = mapped_column(String(128), nullable=False)
    payload_json: Mapped[dict] = mapped_column("payload", JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)


async def get_session(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session
