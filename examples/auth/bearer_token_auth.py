"""Simple Bearer Token authentication example.

Configure in agentseek.json:
    {
        "auth": {
            "path": "./examples/auth/bearer_token_auth.py:auth"
        }
    }

Environment variables:
    AUTH_BEARER_TOKENS=token1=user1,token2=user2
"""

import os

from langgraph_sdk import Auth

VALID_TOKENS: dict[str, str] = {}
raw = os.environ.get("AUTH_BEARER_TOKENS", "")
for entry in raw.split(","):
    entry = entry.strip()
    if "=" in entry:
        token, user_id = entry.split("=", maxsplit=1)
        VALID_TOKENS[token.strip()] = user_id.strip()

auth = Auth()


@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    if not authorization:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid Authorization scheme")

    identity = VALID_TOKENS.get(token)
    if identity is None:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token")
    return {"identity": identity}
