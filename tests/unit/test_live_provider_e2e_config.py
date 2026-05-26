import pytest

from tests.e2e.conftest import should_fail_live_provider_config


@pytest.mark.parametrize(
    ("ci_value", "required_value", "expected"),
    [
        ("true", "", False),
        ("true", "0", False),
        ("true", "false", False),
        ("true", "1", True),
        ("true", "true", True),
        ("false", "true", False),
    ],
)
def test_should_fail_live_provider_config(
    monkeypatch: pytest.MonkeyPatch,
    ci_value: str,
    required_value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("CI", ci_value)
    if required_value:
        monkeypatch.setenv("LIVE_PROVIDER_REQUIRED", required_value)
    else:
        monkeypatch.delenv("LIVE_PROVIDER_REQUIRED", raising=False)

    assert should_fail_live_provider_config() is expected
