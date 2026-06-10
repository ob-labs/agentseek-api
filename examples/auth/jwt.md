# Auth Example

Configure authentication via `agentseek.json`:

```json
{
  "auth": {
    "path": "./auth.py:auth"
  }
}
```

Or via environment variable: `AUTH_MODULE_PATH=./auth.py:auth`

The auth module should export a `langgraph_sdk.Auth` object or a class implementing
the `AuthBackend` protocol (`async def authenticate(self, request) -> User`).

See `custom_backend.py` for a minimal protocol-based example, or `demo/auth.py`
for the `langgraph_sdk.Auth` decorator style.
