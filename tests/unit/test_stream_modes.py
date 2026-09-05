import pytest

from agentseek_api.services.stream_modes import DEFAULT_STREAM_MODES, normalize_stream_modes


def test_none_returns_default() -> None:
    assert normalize_stream_modes(None) == ["values"]


def test_default_is_returned_as_a_fresh_list() -> None:
    # Must not hand out the module-level default for callers to mutate.
    result = normalize_stream_modes(None)
    result.append("messages")
    assert DEFAULT_STREAM_MODES == ["values"]


def test_bare_string_is_wrapped() -> None:
    assert normalize_stream_modes("messages") == ["messages"]


def test_dedupes_preserving_order() -> None:
    assert normalize_stream_modes(["values", "messages", "values"]) == ["values", "messages"]


def test_empty_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported stream_mode"):
        normalize_stream_modes([])


def test_blank_string_entry_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported stream_mode"):
        normalize_stream_modes([""])


def test_unsupported_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="valeus"):
        normalize_stream_modes(["valeus"])


def test_all_supported_modes_pass() -> None:
    modes = ["values", "updates", "messages", "messages-tuple", "debug", "events", "tasks", "checkpoints", "custom"]
    assert normalize_stream_modes(modes) == modes
