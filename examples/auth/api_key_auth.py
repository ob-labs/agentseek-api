"""API Key authentication example.

Configure in agentseek.json:
    {
        "auth": {
            "path": "./examples/auth/api_key_auth.py:auth"
        }
    }

Environment variables:
    AUTH_API_KEYS=key1=user1,key2=user2
"""

import os

from langgraph_sdk import Auth

VALID_KEYS: dict[str, str] = {}
raw = os.environ.get("AUTH_API_KEYS", "")
for entry in raw.split(","):
    entry = entry.strip()
    if "=" in entry:
        key, user_id = entry.split("=", maxsplit=1)
        VALID_KEYS[key.strip()] = user_id.strip()

auth = Auth()


@auth.authenticate
async def get_current_user(headers: dict) -> Auth.types.MinimalUserDict:
    api_key = headers.get(b"x-api-key")
    if not api_key:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Missing X-API-Key header")
    api_key_str = api_key.decode() if isinstance(api_key, bytes) else api_key
    identity = VALID_KEYS.get(api_key_str)
    if identity is None:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid API key")
    return {"identity": identity}
