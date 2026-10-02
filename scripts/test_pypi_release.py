"""Smoke-test an exact published release, never the checkout or a cached wheel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import urlopen
import venv


IMPORT_PROBE = """
import importlib.metadata
from pathlib import Path
import sys
import agentseek_api
import agentseek_api.cli
import agentseek_api.main

expected = sys.argv[1]
assert importlib.metadata.version('agentseek-api') == expected
assert agentseek_api.__version__ == expected
origin = Path(agentseek_api.__file__).resolve()
assert origin.is_relative_to(Path(sys.prefix).resolve()), origin
print(f'Installed import verified: {origin}')
"""


def version_from_tag(tag: str) -> str:
    if not re.fullmatch(r"v\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?", tag):
        raise ValueError(f"Invalid release tag: {tag!r}")
    return tag[1:]


def wait_for_release(version: str, *, attempts: int = 12, interval: float = 10) -> None:
    """Retry publication visibility only; installation/test failures are fatal."""
    url = f"https://pypi.org/pypi/agentseek-api/{version}/json"
    for attempt in range(attempts):
        try:
            with urlopen(url, timeout=15) as response:
                release = json.load(response)
            if release["info"]["version"] != version:
                raise RuntimeError("PyPI returned a different release version")
            if any(file["packagetype"] == "bdist_wheel" and not file.get("yanked", False)
                   for file in release["urls"]):
                return
        except HTTPError as exc:
            if exc.code not in {404, 429, 500, 502, 503, 504}:
                raise
        except (URLError, TimeoutError):
            pass
        if attempt + 1 < attempts:
            print(f"Waiting for agentseek-api {version} on PyPI ({attempt + 1}/{attempts})", flush=True)
            time.sleep(interval)
    raise RuntimeError(f"agentseek-api {version} wheel is not available on PyPI")


def verify_install_report(report: dict, version: str) -> None:
    packages = [item for item in report["install"]
                if item["metadata"]["name"].replace("_", "-").lower() == "agentseek-api"]
    if len(packages) == 1:
        package = packages[0]
        url = urlsplit(package["download_info"]["url"])
        if (package["metadata"]["version"] == version
                and not package.get("is_direct", False)
                and url.scheme == "https" and url.hostname == "files.pythonhosted.org"
                and url.path.endswith(".whl")):
            print(f"PyPI artifact verified: {package['download_info']['url']}", flush=True)
            return
    raise RuntimeError(f"Installation did not use the expected PyPI artifact for {version}")


def check_install(version: str, directory: Path) -> None:
    environment = directory / "venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    binaries = environment / ("Scripts" if sys.platform == "win32" else "bin")
    python = binaries / ("python.exe" if sys.platform == "win32" else "python")
    cli = binaries / ("agentseek-api.exe" if sys.platform == "win32" else "agentseek-api")
    # Drop pip overrides (indexes, targets, constraints, etc.) and Python import
    # overrides. Disable all pip config files, including machine-wide ones.
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(("PIP_", "PYTHON"))}
    env["PIP_CONFIG_FILE"] = os.devnull
    report = directory / "install-report.json"
    common = {"cwd": directory, "env": env, "check": True}
    pip = [str(python), "-I", "-m", "pip", "--disable-pip-version-check"]
    subprocess.run([
        *pip, "--no-cache-dir", "install", "--index-url", "https://pypi.org/simple",
        "--only-binary=agentseek-api", "--report", str(report), f"agentseek-api=={version}",
    ], **common, timeout=600)
    verify_install_report(json.loads(report.read_text(encoding="utf-8")), version)
    subprocess.run([*pip, "check"], **common, timeout=60)
    subprocess.run([str(python), "-I", "-c", IMPORT_PROBE, version], **common, timeout=60)
    result = subprocess.run([str(cli), "version"], **common, timeout=60, capture_output=True, text=True)
    if result.stdout.strip() != f"agentseek-api {version}":
        raise RuntimeError(f"Unexpected installed CLI version: {result.stdout!r}")
    print(result.stdout.strip(), flush=True)
    subprocess.run([str(cli), "--help"], **common, timeout=60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=os.environ.get("RELEASE_TAG"), help="Exact published tag, e.g. v0.3.3")
    args = parser.parse_args()
    if not args.tag:
        parser.error("--tag or RELEASE_TAG is required")
    version = version_from_tag(args.tag)
    wait_for_release(version)
    with tempfile.TemporaryDirectory(prefix="agentseek-pypi-") as directory:
        check_install(version, Path(directory))
    print(f"PyPI release smoke passed: agentseek-api {version} on {sys.platform}", flush=True)


if __name__ == "__main__":
    main()
