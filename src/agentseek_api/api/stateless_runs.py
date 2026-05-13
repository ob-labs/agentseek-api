from fastapi import APIRouter, Depends

from agentseek_api.core.auth_deps import get_current_user
from agentseek_api.models.api import RunCreate, RunRead, ThreadCreate
from agentseek_api.models.auth import User
from agentseek_api.services.thread_service import create_thread_for_user
from agentseek_api.api.runs import create_run

router = APIRouter(prefix="/runs", tags=["Stateless Runs"])


@router.post("", response_model=RunRead)
async def create_stateless_run(payload: RunCreate, user: User = Depends(get_current_user)) -> RunRead:
    thread = await create_thread_for_user(payload=ThreadCreate(metadata={"stateless": True}), user=user)
    return await create_run(thread.thread_id, payload, user)
