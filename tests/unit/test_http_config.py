from pathlib import Path

from agentseek_api.core.http_config import get_config_dir, get_http_config
from agentseek_api.settings import settings


def test_get_http_config_returns_none_without_config(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text('{"graphs":{}}', encoding="utf-8")
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))
    assert get_http_config() is None


def test_get_http_config_returns_http_section(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text(
        '{"graphs":{}, "http":{"app":"./custom.py:app"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))
    result = get_http_config()
    assert result is not None
    assert result["app"] == "./custom.py:app"


def test_get_config_dir(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "agentseek.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", str(config_path))
    assert get_config_dir() == tmp_path.resolve()


def test_get_config_dir_returns_none_without_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "AGENTSEEK_GRAPHS", "")
    monkeypatch.chdir(tmp_path)
    assert get_config_dir() is None
