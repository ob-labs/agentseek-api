"""Shared dotenv interpolation cases for supported-version and handoff tests."""

from __future__ import annotations

import importlib.metadata
import os
import tempfile
from pathlib import Path

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

DOTENV_CONFORMANCE_ENV_KEYS = frozenset(
    {
        "PR69_MISSING",
        "PR69_DISALLOWED_SECRET",
        "OPENAI_ALLOWED_SOURCE",
        "CONF_ORIGIN",
        "A.B",
        "1LEADING",
        "CONF_EMPTY",
        "CONF_VALUELESS",
    }
)

CROSS_LAYER_TOMBSTONE_CASES = (
    {
        "name": "config-dotenv",
        "config_env": "./config.env",
        "config_dotenv": "CONF_TOMBSTONE=from-config-dotenv\n",
        "shell_env": {},
        "expected": {},
    },
    {
        "name": "config-mapping",
        "config_env": {"CONF_TOMBSTONE": "from-config-mapping"},
        "config_dotenv": None,
        "shell_env": {},
        "expected": {},
    },
    {
        "name": "allowlisted-shell-restores",
        "config_env": {"OPENAI_API_KEY": "from-config"},
        "config_dotenv": None,
        "shell_env": {"OPENAI_API_KEY": "from-shell"},
        "expected": {"OPENAI_API_KEY": "from-shell"},
    },
)


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
            for index, case in enumerate(DOTENV_CONFORMANCE_CASES):
                for key in DOTENV_CONFORMANCE_ENV_KEYS:
                    os.environ.pop(key, None)
                os.environ.update(case["ambient"])
                env_file = root / f"{index}.env"
                env_file.write_text(case["contents"], encoding="utf-8")
                upstream = dotenv_values(env_file)
                expected = {key: value for key, value in upstream.items() if value is not None}
                actual = build_runtime_env(
                    config_path=None,
                    env_file=str(env_file),
                    cwd=root,
                    base_env=dict(os.environ),
                )
                assert {key: actual[key] for key in expected} == expected, case["name"]
                assert all(key not in actual for key, value in upstream.items() if value is None), case["name"]
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
