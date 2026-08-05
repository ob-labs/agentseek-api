from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from agentseek_api.api.runs import _validate_supported_run_controls


def test_validate_rejects_configurable_and_context_together() -> None:
    payload = SimpleNamespace(
        config={"configurable": {"tenant": "acme"}},
        context={"org": "ob"},
    )
    with pytest.raises(HTTPException) as excinfo:
        _validate_supported_run_controls(payload, stateless=False)
    assert excinfo.value.status_code == 400


def test_validate_accepts_configurable_or_context_alone() -> None:
    _validate_supported_run_controls(
        SimpleNamespace(config={"configurable": {"tenant": "acme"}}, context={}),
        stateless=False,
    )
    _validate_supported_run_controls(
        SimpleNamespace(config={}, context={"org": "ob"}),
        stateless=False,
    )
