from agentseek_api.core.database import db_manager
from agentseek_api.core.orm import Thread
from agentseek_api.models.api import ThreadCreate, ThreadRead
from agentseek_api.models.auth import User


async def create_thread_for_user(*, payload: ThreadCreate, user: User) -> ThreadRead:
    session_factory = db_manager.get_session_factory()
    async with session_factory() as session:
        row = Thread(user_id=user.identity, metadata_json=payload.metadata)
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return ThreadRead(
            thread_id=row.thread_id,
            user_id=row.user_id,
            metadata=row.metadata_json,
            created_at=row.created_at,
            updated_at=row.updated_at,
            state_updated_at=row.state_updated_at,
            config=row.config_json,
            status=row.status,
        )
