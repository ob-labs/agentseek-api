from langgraph_sdk import Auth

HeaderAuthBackend = Auth()


@HeaderAuthBackend.authenticate
async def authenticate(headers: dict[bytes, bytes]) -> dict:
    raw = headers.get(b"x-user-id", b"default_user")
    identity = raw.decode() if isinstance(raw, bytes) else str(raw)
    return {"identity": identity}


@HeaderAuthBackend.on.threads.create
async def on_threads_create(ctx: Auth.types.AuthContext, value: dict) -> None:
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity


@HeaderAuthBackend.on.threads.read
async def on_threads_read(ctx: Auth.types.AuthContext, value: dict) -> dict:
    return {"owner": ctx.user.identity}


@HeaderAuthBackend.on.threads.update
async def on_threads_update(ctx: Auth.types.AuthContext, value: dict) -> dict:
    return {"owner": ctx.user.identity}


@HeaderAuthBackend.on.threads.delete
async def on_threads_delete(ctx: Auth.types.AuthContext, value: dict) -> dict:
    return {"owner": ctx.user.identity}


@HeaderAuthBackend.on.threads.search
async def on_threads_search(ctx: Auth.types.AuthContext, value: dict) -> dict:
    return {"owner": ctx.user.identity}
