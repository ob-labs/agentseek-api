from pathlib import Path

from agentseek_api.core.a2a_config import is_a2a_enabled
from agentseek_api.settings import settings


def test_is_a2a_enabled_defaults_true_without_http_section(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text('{"graphs":{"chat":"chat.graph:graph"}}', encoding="utf-8")
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    assert is_a2a_enabled() is True


def test_is_a2a_enabled_respects_disable_flag(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "http": {
    "disable_a2a": true
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    assert is_a2a_enabled() is False


def test_is_a2a_enabled_fails_closed_for_invalid_config(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "http": {
    "disable_a2a": true
  }
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    assert is_a2a_enabled() is False


def test_is_a2a_enabled_fails_closed_for_invalid_http_section(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "http": []
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    assert is_a2a_enabled() is False


def test_is_a2a_enabled_fails_closed_for_invalid_disable_flag(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        """
{
  "graphs": {
    "chat": "chat.graph:graph"
  },
  "http": {
    "disable_a2a": "true"
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))

    assert is_a2a_enabled() is False
