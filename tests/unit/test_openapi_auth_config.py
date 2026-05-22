from pathlib import Path

from agentseek_api.main import create_app
from agentseek_api.settings import settings


def test_create_app_merges_auth_openapi_from_json_config(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        """
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "auth": {
    "path": "./auth.py:auth",
    "openapi": {
      "securitySchemes": {
        "apiKeyAuth": {
          "type": "apiKey",
          "in": "header",
          "name": "X-API-Key"
        }
      },
      "security": [{ "apiKeyAuth": [] }]
    },
    "disable_studio_auth": false
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    schema = create_app().openapi()

    assert schema["components"]["securitySchemes"]["apiKeyAuth"] == {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
    }
    assert schema["security"] == [{"apiKeyAuth": []}]
    assert "security" not in schema["paths"]["/assistants"]["post"]
    assert "security" not in schema["paths"]["/assistants"]["get"]
    assert "security" not in schema["paths"]["/agents"]["post"]
    assert "security" not in schema["paths"]["/agents"]["get"]
