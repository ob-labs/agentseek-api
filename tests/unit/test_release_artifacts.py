from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_release_build_excludes_private_superpowers_artifacts() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    assert "/.superpowers" in project["tool"]["hatch"]["build"]["exclude"]
    assert "/docs/superpowers" in project["tool"]["hatch"]["build"]["exclude"]
