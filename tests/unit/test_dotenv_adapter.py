from __future__ import annotations

from pathlib import Path

import pytest


def test_parse_dotenv_file_preserves_file_local_physical_order(
    tmp_path: Path,
) -> None:
    from agentseek_api.dotenv_adapter import parse_dotenv_file

    env_file = tmp_path / "runtime.env"
    env_file.write_text(
        "export FIRST=one\n"
        "DUPLICATE=first\n"
        "FROM_DUPLICATE=${DUPLICATE}/v1\n"
        "DUPLICATE=second\n"
        'MULTILINE="line one\nline two"\n'
        "FROM_AMBIENT=${AMBIENT}/v2\n"
        "BARE_REFERENCE=$AMBIENT\n"
        "MISSING_REFERENCE=${UNSET}\n"
        "MISSING_DEFAULT=${UNSET:-fallback}\n"
        "EMPTY=\n"
        "VALUELESS\n"
        "FROM_VALUELESS=${VALUELESS:-fallback}\n",
        encoding="utf-8",
    )
    ambient = {"AMBIENT": "from-shell"}

    values = parse_dotenv_file(env_file, ambient=ambient)

    assert values == {
        "FIRST": "one",
        "DUPLICATE": "second",
        "FROM_DUPLICATE": "first/v1",
        "MULTILINE": "line one\nline two",
        "FROM_AMBIENT": "from-shell/v2",
        "BARE_REFERENCE": "$AMBIENT",
        "MISSING_REFERENCE": "",
        "MISSING_DEFAULT": "fallback",
        "EMPTY": "",
        "VALUELESS": None,
        "FROM_VALUELESS": "",
    }
    assert ambient == {"AMBIENT": "from-shell"}


@pytest.mark.parametrize(
    "contents",
    [
        'BROKEN "value"\n',
        'UNTERMINATED="value\n',
    ],
)
def test_parse_dotenv_file_rejects_genuinely_malformed_syntax(
    tmp_path: Path,
    contents: str,
) -> None:
    from agentseek_api.dotenv_adapter import DotenvFileError, parse_dotenv_file

    env_file = tmp_path / "broken.env"
    env_file.write_text("SECRET=must-not-leak\n" + contents, encoding="utf-8")

    with pytest.raises(DotenvFileError) as raised:
        parse_dotenv_file(env_file, ambient={})

    assert raised.value.path == env_file
    assert raised.value.line == 2
    assert "must-not-leak" not in str(raised.value)


def test_parse_dotenv_file_reports_missing_source(tmp_path: Path) -> None:
    from agentseek_api.dotenv_adapter import DotenvFileError, parse_dotenv_file

    env_file = tmp_path / "missing.env"

    with pytest.raises(DotenvFileError, match="does not exist"):
        parse_dotenv_file(env_file, ambient={})


def test_parse_dotenv_file_reports_utf8_decode_failure(tmp_path: Path) -> None:
    from agentseek_api.dotenv_adapter import DotenvFileError, parse_dotenv_file

    env_file = tmp_path / "invalid.env"
    env_file.write_bytes(b"TOKEN=\xff\n")

    with pytest.raises(DotenvFileError, match="not valid UTF-8"):
        parse_dotenv_file(env_file, ambient={})
