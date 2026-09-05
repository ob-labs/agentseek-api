from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from pymysql.err import OperationalError

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "seekdb_checkpoint_smoke.py"
SPEC = spec_from_file_location("seekdb_checkpoint_smoke", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
seekdb_checkpoint_smoke = module_from_spec(SPEC)
SPEC.loader.exec_module(seekdb_checkpoint_smoke)


@pytest.mark.parametrize("error_code", [4012, 4392])
def test_retry_transient_timeout_retries_known_oceanbase_errors(
    monkeypatch: pytest.MonkeyPatch, error_code: int
) -> None:
    attempts = {"count": 0}
    monkeypatch.setattr(seekdb_checkpoint_smoke.time, "sleep", lambda _seconds: None)

    def flaky() -> None:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise OperationalError(error_code, "transient")

    seekdb_checkpoint_smoke._retry_transient_timeout("checkpoint setup", flaky, timeout_seconds=5.0)

    assert attempts["count"] == 3


def test_retry_transient_timeout_raises_non_retryable_operational_error() -> None:
    def fail() -> None:
        raise OperationalError(9999, "boom")

    with pytest.raises(OperationalError, match="boom"):
        seekdb_checkpoint_smoke._retry_transient_timeout("checkpoint setup", fail, timeout_seconds=5.0)
