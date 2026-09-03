from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_release_build_excludes_private_superpowers_artifacts() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    assert "/.superpowers" in project["tool"]["hatch"]["build"]["exclude"]
    assert "/docs/superpowers" in project["tool"]["hatch"]["build"]["exclude"]


def test_readme_badges_state_the_bounded_supported_python_range() -> None:
    badge = "[![Python 3.12-3.13](https://img.shields.io/badge/python-3.12--3.13-blue.svg)]"
    old_unbounded_badge = "https://img.shields.io/badge/python-%3E%3D3.12-blue.svg"

    for readme_name in ("README.md", "README.zh-CN.md"):
        readme = (ROOT / readme_name).read_text(encoding="utf-8")
        assert badge in readme
        assert old_unbounded_badge not in readme
