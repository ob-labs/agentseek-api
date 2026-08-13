"""Shared dotenv interpolation cases for supported-version and handoff tests."""

from __future__ import annotations

import importlib.metadata
import io
import json
import os
import tempfile
from pathlib import Path

from dotenv.parser import parse_stream
from dotenv.variables import Variable, parse_variables

DOTENV_CONFORMANCE_CASES = (
    {
        "name": "missing",
        "contents": "CONF_MISSING=prefix-${PR69_MISSING}-suffix\n",
        "ambient": {},
        "container_expected": {"CONF_MISSING": "prefix-${PR69_MISSING}-suffix"},
        "container_absent": (),
    },
    {
        "name": "duplicate-order",
        "contents": (
            "CONF_ORIGIN=https://first.example\n"
            "CONF_ORDERED=${CONF_ORIGIN}/v1\n"
            "CONF_ORIGIN=https://second.example\n"
        ),
        "ambient": {},
        "container_expected": {
            "CONF_ORIGIN": "https://second.example",
            "CONF_ORDERED": "https://first.example/v1",
        },
        "container_absent": (),
    },
    {
        "name": "broad-names",
        "contents": "A.B=dotted\n1LEADING=digit\nCONF_BROAD=${A.B}-${1LEADING}\n",
        "ambient": {},
        "container_expected": {"CONF_BROAD": "dotted-digit"},
        "container_absent": (),
    },
    {
        "name": "multiline-default",
        "contents": 'CONF_MULTILINE="${PR69_MISSING:-first line\nsecond line}"\n',
        "ambient": {},
        "container_expected": {"CONF_MULTILINE": "first line\nsecond line"},
        "container_absent": (),
    },
    {
        "name": "bare-default",
        "contents": "CONF_BARE=$PR69_MISSING\nCONF_DEFAULT=${PR69_MISSING:-fallback}\n",
        "ambient": {},
        "container_expected": {
            "CONF_BARE": "$PR69_MISSING",
            "CONF_DEFAULT": "fallback",
        },
        "container_absent": (),
    },
    {
        "name": "empty-valueless",
        "contents": (
            "CONF_EMPTY=\n"
            "CONF_VALUELESS\n"
            "CONF_FROM_EMPTY=${CONF_EMPTY:-fallback}\n"
            "CONF_FROM_VALUELESS=${CONF_VALUELESS:-fallback}\n"
        ),
        "ambient": {},
        "container_expected": {
            "CONF_EMPTY": "",
            "CONF_FROM_EMPTY": "",
            "CONF_FROM_VALUELESS": "",
        },
        "container_absent": ("CONF_VALUELESS",),
    },
    {
        "name": "allowed-disallowed-ambient",
        "contents": (
            "CONF_ALLOWED=${OPENAI_ALLOWED_SOURCE}\n"
            "CONF_DISALLOWED=${PR69_DISALLOWED_SECRET}\n"
        ),
        "ambient": {
            "OPENAI_ALLOWED_SOURCE": "allowlisted-source",
            "PR69_DISALLOWED_SECRET": "host-sensitive-value",
        },
        "container_expected": {
            "CONF_ALLOWED": "allowlisted-source",
            "CONF_DISALLOWED": "${PR69_DISALLOWED_SECRET}",
        },
        "container_absent": ("PR69_DISALLOWED_SECRET",),
    },
)

CROSS_LAYER_CONFORMANCE_CASES = (
    {
        "name": "config-dotenv-reference",
        "config_env": "./config.env",
        "config_dotenv": "CONF_SOURCE=https://dotenv.example\n",
        "cli_dotenv": "CONF_RESULT=${CONF_SOURCE}/v1\n",
        "shell_env": {},
        "expected": {
            "CONF_SOURCE": "https://dotenv.example",
            "CONF_RESULT": "https://dotenv.example/v1",
        },
        "absent": (),
    },
    {
        "name": "config-mapping-reference",
        "config_env": {"CONF_SOURCE": "https://mapping.example"},
        "config_dotenv": None,
        "cli_dotenv": "CONF_RESULT=${CONF_SOURCE}/v1\n",
        "shell_env": {},
        "expected": {
            "CONF_SOURCE": "https://mapping.example",
            "CONF_RESULT": "https://mapping.example/v1",
        },
        "absent": (),
    },
    {
        "name": "final-allowlisted-shell-override",
        "config_env": {"OPENAI_API_KEY": "from-config"},
        "config_dotenv": None,
        "cli_dotenv": "CONF_RESULT=${OPENAI_API_KEY}\n",
        "shell_env": {"OPENAI_API_KEY": "from-shell"},
        "expected": {
            "CONF_RESULT": "from-config",
            "OPENAI_API_KEY": "from-shell",
        },
        "absent": (),
    },
    {
        "name": "config-dotenv-valueless",
        "config_env": "./config.env",
        "config_dotenv": "CONF_TOMBSTONE=from-config-dotenv\n",
        "cli_dotenv": "CONF_TOMBSTONE\nCONF_RESULT=${CONF_TOMBSTONE:-fallback}\n",
        "shell_env": {},
        "expected": {
            "CONF_TOMBSTONE": "from-config-dotenv",
            "CONF_RESULT": "",
        },
        "absent": (),
    },
    {
        "name": "config-mapping-valueless",
        "config_env": {"CONF_TOMBSTONE": "from-config-mapping"},
        "config_dotenv": None,
        "cli_dotenv": "CONF_TOMBSTONE\nCONF_RESULT=${CONF_TOMBSTONE:-fallback}\n",
        "shell_env": {},
        "expected": {
            "CONF_TOMBSTONE": "from-config-mapping",
            "CONF_RESULT": "",
        },
        "absent": (),
    },
    {
        "name": "config-dotenv-valueless-does-not-mask-shell-for-next-file",
        "config_env": "./config.env",
        "config_dotenv": "OPENAI_API_KEY\n",
        "cli_dotenv": "CONF_RESULT=${OPENAI_API_KEY}\n",
        "shell_env": {"OPENAI_API_KEY": "from-shell"},
        "expected": {
            "OPENAI_API_KEY": "from-shell",
            "CONF_RESULT": "from-shell",
        },
        "absent": (),
    },
)

CONFORMANCE_AMBIENT_MODES = (
    ("clean", {}),
    (
        "hostile",
        {
            "OPENAI_API_KEY": "ambient-provider-key",
            "OPENAI_ALLOWED_SOURCE": "ambient-allowed-source",
            "PR69_MISSING": "ambient-missing-value",
            "PR69_DISALLOWED_SECRET": "host-sensitive-value",
            "A.B": "ambient-dotted-value",
            "1LEADING": "ambient-digit-value",
        },
    ),
)


def _dotenv_keys(contents: str) -> set[str]:
    keys: set[str] = set()
    for binding in parse_stream(io.StringIO(contents)):
        if binding.key is not None:
            keys.add(binding.key)
        if binding.value is not None:
            keys.update(atom.name for atom in parse_variables(binding.value) if isinstance(atom, Variable))
    return keys


def _conformance_env_keys() -> frozenset[str]:
    keys: set[str] = set()
    for case in DOTENV_CONFORMANCE_CASES:
        keys.update(_dotenv_keys(case["contents"]))
        keys.update(case["ambient"])
        keys.update(case["container_expected"])
        keys.update(case["container_absent"])
    for case in CROSS_LAYER_CONFORMANCE_CASES:
        config_env = case["config_env"]
        if isinstance(config_env, dict):
            keys.update(config_env)
        if case["config_dotenv"] is not None:
            keys.update(_dotenv_keys(case["config_dotenv"]))
        keys.update(_dotenv_keys(case["cli_dotenv"]))
        keys.update(case["shell_env"])
        keys.update(case["expected"])
        keys.update(case["absent"])
    return frozenset(keys)


DOTENV_CONFORMANCE_ENV_KEYS = _conformance_env_keys()


def assert_runtime_conformance(*, expected_dotenv_version: str | None = None) -> None:
    """Compare the runtime loader with the installed python-dotenv version."""
    from dotenv import dotenv_values

    from agentseek_api.cli import build_runtime_env

    if expected_dotenv_version is not None:
        assert importlib.metadata.version("python-dotenv") == expected_dotenv_version

    previous = {key: os.environ.get(key) for key in DOTENV_CONFORMANCE_ENV_KEYS}
    try:
        with tempfile.TemporaryDirectory(prefix="agentseek-dotenv-") as directory:
            root = Path(directory)
            for mode_name, mode_ambient in CONFORMANCE_AMBIENT_MODES:
                for index, case in enumerate(DOTENV_CONFORMANCE_CASES):
                    for key in DOTENV_CONFORMANCE_ENV_KEYS:
                        os.environ.pop(key, None)
                    os.environ.update(mode_ambient)
                    os.environ.update(case["ambient"])
                    env_file = root / f"{mode_name}-{index}.env"
                    env_file.write_text(case["contents"], encoding="utf-8")
                    upstream = dotenv_values(env_file)
                    expected = {key: value for key, value in upstream.items() if value is not None}
                    expected.update({key: os.environ[key] for key in upstream.keys() & os.environ.keys()})
                    actual = build_runtime_env(
                        config_path=None,
                        env_file=str(env_file),
                        cwd=root,
                        base_env=dict(os.environ),
                    )
                    assertion = f"{mode_name}/{case['name']}"
                    assert {key: actual[key] for key in expected} == expected, assertion
                    assert all(
                        key not in actual
                        for key, value in upstream.items()
                        if value is None and key not in os.environ
                    ), assertion

                for index, case in enumerate(CROSS_LAYER_CONFORMANCE_CASES):
                    for key in DOTENV_CONFORMANCE_ENV_KEYS:
                        os.environ.pop(key, None)
                    os.environ.update(mode_ambient)
                    os.environ.update(case["shell_env"])
                    config_path = root / f"cross-layer-{mode_name}-{index}.json"
                    config_path.write_text(
                        json.dumps({"graphs": {"chat": "chat.graph:graph"}, "env": case["config_env"]}),
                        encoding="utf-8",
                    )
                    if case["config_dotenv"] is not None:
                        (root / "config.env").write_text(case["config_dotenv"], encoding="utf-8")
                    env_file = root / f"cross-layer-{mode_name}-{index}.env"
                    env_file.write_text(case["cli_dotenv"], encoding="utf-8")
                    actual = build_runtime_env(
                        config_path=config_path,
                        env_file=str(env_file),
                        cwd=root,
                        base_env=dict(os.environ),
                    )
                    assertion = f"{mode_name}/{case['name']}"
                    assert {key: actual[key] for key in case["expected"]} == case["expected"], assertion
                    assert all(key not in actual for key in case["absent"]), assertion
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
