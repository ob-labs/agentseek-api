"""Release smoke contracts: public artifacts, clean installs, and hard failures."""

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError
import venv

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/test_pypi_release.py"


def test_smoke_script_exposes_usage_without_installing():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--tag" in result.stdout


@pytest.fixture
def smoke():
    assert SCRIPT.is_file(), "the portable PyPI release smoke helper is missing"
    spec = importlib.util.spec_from_file_location("pypi_release_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(version="0.3.3", url="https://files.pythonhosted.org/packages/example.whl"):
    return {"install": [{
        "metadata": {"name": "agentseek-api", "version": version},
        "download_info": {"url": url}, "is_direct": False,
    }]}


def test_release_smoke_matrix_waits_for_publication_and_runs_every_os():
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    job = workflow["jobs"].get("pypi-smoke")
    assert job is not None, "release workflow has no post-publication smoke job"
    assert job["needs"] == "publish-pypi"
    assert set(job["strategy"]["matrix"]["os"]) == {
        "ubuntu-latest", "macos-latest", "windows-latest",
    }
    assert job["strategy"]["fail-fast"] is False
    assert job["runs-on"] == "${{ matrix.os }}"
    assert job.get("continue-on-error", False) is False
    assert job["permissions"] == {"contents": "read"}
    command = next(step for step in job["steps"] if "run" in step)
    assert command["env"]["RELEASE_TAG"] == "${{ github.ref_name }}"
    assert command["run"].split() == ["python", "scripts/test_pypi_release.py"]


@pytest.mark.parametrize("tag", ["v0.3.3", "v1.0.0rc1"])
def test_exact_version_is_derived_from_release_tag(smoke, tag):
    assert smoke.version_from_tag(tag) == tag[1:]


@pytest.mark.parametrize("tag", ["", "0.3.3", "v", "v0.3.3;echo bad", "v../0.3.3"])
def test_invalid_tag_cannot_become_a_pip_requirement(smoke, tag):
    with pytest.raises(ValueError, match="release tag"):
        smoke.version_from_tag(tag)


def test_waits_for_visible_wheel_after_publication(smoke, monkeypatch):
    replies = iter([
        HTTPError("https://pypi.org", 404, "not yet", {}, None),
        {"info": {"version": "0.3.3"}, "urls": []},
        {"info": {"version": "0.3.3"}, "urls": [{"packagetype": "bdist_wheel", "yanked": False}]},
    ])
    urls, delays = [], []

    def request(url, **kwargs):
        urls.append(url)
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(smoke, "urlopen", request)
    monkeypatch.setattr(smoke.time, "sleep", delays.append)
    smoke.wait_for_release("0.3.3", attempts=3, interval=1)
    assert urls == ["https://pypi.org/pypi/agentseek-api/0.3.3/json"] * 3
    assert delays == [1, 1]


def test_publication_wait_is_bounded_and_never_falls_back_to_latest(smoke, monkeypatch):
    def unavailable(*args, **kwargs):
        raise HTTPError("https://pypi.org", 404, "missing", {}, None)

    delays = []
    monkeypatch.setattr(smoke, "urlopen", unavailable)
    monkeypatch.setattr(smoke.time, "sleep", delays.append)
    with pytest.raises(RuntimeError, match="0.3.3.*not available"):
        smoke.wait_for_release("0.3.3", attempts=3, interval=1)
    assert delays == [1, 1]


@pytest.mark.parametrize("report", [
    _report(version="0.3.2"),
    _report(url="file:///checkout/dist/agentseek_api.whl"),
    _report(url="https://private-index.example/agentseek_api.whl"),
    {"install": []},
])
def test_install_report_rejects_wrong_version_or_non_pypi_artifact(smoke, report):
    with pytest.raises(RuntimeError, match="PyPI artifact"):
        smoke.verify_install_report(report, "0.3.3")


def test_clean_install_uses_pinned_public_wheel_and_propagates_cli_failure(smoke, monkeypatch, tmp_path):
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://private-index.example/simple")
    monkeypatch.setenv("PIP_TARGET", str(ROOT / "src"))
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    monkeypatch.setattr(smoke.venv.EnvBuilder, "create", lambda self, path: Path(path).mkdir())
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "--report" in command:
            Path(command[command.index("--report") + 1]).write_text(json.dumps(_report()))
        if command[-1] == "version":
            raise subprocess.CalledProcessError(2, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(smoke.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        smoke.check_install("0.3.3", tmp_path)
    install = next(command for command, _ in calls if "--report" in command)
    assert install[-1] == "agentseek-api==0.3.3"
    assert install[install.index("--index-url") + 1] == "https://pypi.org/simple"
    assert "--no-cache-dir" in install and "--only-binary=agentseek-api" in install
    for command, kwargs in calls:
        assert kwargs["check"] is True
        assert kwargs["cwd"] == tmp_path
        assert not any(key.startswith("PYTHON") for key in kwargs["env"])
        assert {key for key in kwargs["env"] if key.startswith("PIP_")} == {"PIP_CONFIG_FILE"}
        assert kwargs["env"]["PIP_CONFIG_FILE"] == os.devnull
    executable = calls[-1][0][0]
    assert executable.endswith("agentseek-api.exe" if sys.platform == "win32" else "agentseek-api")


@pytest.mark.parametrize("metadata_version, module_version, external, expected_success", [
    ("0.3.3", "0.3.3", False, True),
    ("0.3.2", "0.3.3", False, False),
    ("0.3.3", "0.3.2", False, False),
    ("0.3.3", "0.3.3", True, False),
])
def test_import_probe_checks_version_and_rejects_external_source(
    smoke, tmp_path, metadata_version, module_version, external, expected_success,
):
    environment = tmp_path / "env"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    site_packages = Path(subprocess.check_output(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True,
    ).strip())
    source = tmp_path / "checkout" if external else site_packages
    source.mkdir(exist_ok=True)
    if external:
        (site_packages / "editable.pth").write_text(str(source))
    package = source / "agentseek_api"
    package.mkdir()
    (package / "__init__.py").write_text(f'__version__ = "{module_version}"\n')
    (package / "main.py").write_text("")
    (package / "cli.py").write_text("")
    metadata = site_packages / f"agentseek_api-{metadata_version}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Name: agentseek-api\nVersion: {metadata_version}\n")
    result = subprocess.run(
        [str(python), "-I", "-c", smoke.IMPORT_PROBE, "0.3.3"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert (result.returncode == 0) is expected_success, result.stderr
