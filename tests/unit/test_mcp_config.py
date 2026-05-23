from pathlib import Path

from agentseek_api.core.config_file import get_active_config_payload
from agentseek_api.core.mcp_config import is_mcp_enabled
from agentseek_api.settings import settings


def test_get_active_config_payload_reads_http_section(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "http": {
    "disable_mcp": true
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    payload = get_active_config_payload()

    assert payload is not None
    assert payload["http"] == {"disable_mcp": True}


def test_is_mcp_enabled_defaults_true_without_http_section(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text('{"graphs":{"chat":"chat.graph:graph"}}', encoding="utf-8")
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    assert is_mcp_enabled() is True


def test_is_mcp_enabled_respects_disable_flag(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "http": {
    "disable_mcp": true
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    assert is_mcp_enabled() is False


def test_is_mcp_enabled_fails_closed_for_invalid_config(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "http": {
    "disable_mcp": true
  }
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    assert get_active_config_payload() is None
    assert is_mcp_enabled() is False
