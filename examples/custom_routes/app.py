from fastapi import APIRouter

from agentseek_api.main import create_app


app = create_app()
router = APIRouter()


@router.get("/custom/ping")
async def custom_ping() -> dict[str, bool]:
    return {"ok": True}


app.include_router(router)
