"""JWT (HS256) authentication example.

Configure in agentseek.json:
    {
        "auth": {
            "path": "./examples/auth/jwt_auth.py:auth"
        }
    }

Environment variables:
    AUTH_JWT_SECRET=your-shared-secret
    AUTH_JWT_ALGORITHM=HS256  (optional, default HS256)
"""

import base64
import hashlib
import hmac
import json
import os
import time

from langgraph_sdk import Auth

JWT_SECRET = os.environ.get("AUTH_JWT_SECRET", "")
JWT_ALGORITHM = os.environ.get("AUTH_JWT_ALGORITHM", "HS256")

auth = Auth()


def _decode_urlsafe_json(segment: str) -> dict | None:
    try:
        padded = segment + ("=" * (-len(segment) % 4))
        raw = base64.urlsafe_b64decode(padded.encode())
        return json.loads(raw)
    except Exception:
        return None


def _verify_hs256(token: str) -> dict | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    header_seg, payload_seg, sig_seg = parts

    header = _decode_urlsafe_json(header_seg)
    if not header or header.get("alg") != JWT_ALGORITHM:
        return None

    signing_input = f"{header_seg}.{payload_seg}".encode()
    expected = hmac.new(JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
    expected_seg = base64.urlsafe_b64encode(expected).rstrip(b"=").decode()
    if not hmac.compare_digest(sig_seg, expected_seg):
        return None

    payload = _decode_urlsafe_json(payload_seg)
    if not payload:
        return None

    now = time.time()
    if "exp" in payload and now >= payload["exp"]:
        return None
    if "nbf" in payload and now < payload["nbf"]:
        return None

    return payload


@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    if not authorization:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid Authorization scheme")
    if not JWT_SECRET:
        raise Auth.exceptions.HTTPException(status_code=500, detail="AUTH_JWT_SECRET not configured")

    payload = _verify_hs256(token)
    if payload is None:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid or expired token")

    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Token missing sub claim")

    return {"identity": sub}
