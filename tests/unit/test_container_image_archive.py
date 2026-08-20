from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


SCANNER = Path("scripts/container_image_archive.py")


def _load_scanner():
    if not SCANNER.is_file():
        pytest.fail("the structural image archive scanner is missing")
    spec = importlib.util.spec_from_file_location("container_image_archive", SCANNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tar(entries: list[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


def _docker_save_archive(
    *,
    layer_payloads: list[bytes],
    config: bytes = b'{"history":[]}',
    config_reference: str = "config.json",
    extra_entries: list[tuple[str, bytes]] | None = None,
) -> bytes:
    layers = [f"layer-{index}/layer.tar" for index in range(len(layer_payloads))]
    manifest = json.dumps(
        [
            {
                "Config": config_reference,
                "RepoTags": ["synthetic:test"],
                "Layers": layers,
            }
        ],
        separators=(",", ":"),
    ).encode()
    entries = [("manifest.json", manifest), ("config.json", config)]
    entries.extend(zip(layers, layer_payloads, strict=True))
    entries.extend(extra_entries or [])
    return _tar(entries)


def _descriptor(payload: bytes, media_type: str) -> dict[str, object]:
    return {
        "digest": f"sha256:{hashlib.sha256(payload).hexdigest()}",
        "mediaType": media_type,
        "size": len(payload),
    }


def _oci_archive(*, layer: bytes, config: bytes = b'{"history":[]}') -> bytes:
    config_descriptor = _descriptor(config, "application/vnd.oci.image.config.v1+json")
    layer_descriptor = _descriptor(layer, "application/vnd.oci.image.layer.v1.tar+gzip")
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "config": config_descriptor,
            "layers": [layer_descriptor],
        },
        separators=(",", ":"),
    ).encode()
    manifest_descriptor = _descriptor(
        manifest, "application/vnd.oci.image.manifest.v1+json"
    )
    index = json.dumps(
        {"schemaVersion": 2, "manifests": [manifest_descriptor]},
        separators=(",", ":"),
    ).encode()
    blobs = [(config_descriptor, config), (layer_descriptor, layer)]
    blobs.append((manifest_descriptor, manifest))
    return _tar(
        [
            ("oci-layout", b'{"imageLayoutVersion":"1.0.0"}'),
            ("index.json", index),
            *[
                (f"blobs/sha256/{descriptor['digest'][7:]}", payload)
                for descriptor, payload in blobs
            ],
        ]
    )


def test_scanner_accepts_every_referenced_layer_and_no_trunc_history() -> None:
    scanner = _load_scanner()
    layers = [
        _tar([("first.txt", b"first-safe-payload")]),
        gzip.compress(_tar([("second.txt", b"second-safe-payload")])),
    ]

    scanner.scan_image_archive(
        _docker_save_archive(layer_payloads=layers),
        forbidden=b"high-entropy-canary",
        history=b'{"CreatedBy":"safe"}\n',
    )


def test_scanner_accepts_an_oci_index_and_verifies_referenced_blobs() -> None:
    scanner = _load_scanner()

    scanner.scan_image_archive(
        _oci_archive(layer=gzip.compress(_tar([("safe", b"safe")]))),
        forbidden=b"high-entropy-canary",
        history=b'{"CreatedBy":"safe"}\n',
    )


def test_scanner_rejects_an_oci_blob_digest_mismatch() -> None:
    scanner = _load_scanner()
    archive = _oci_archive(layer=gzip.compress(_tar([("safe", b"safe")])))

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        entries = [
            (member.name, source.extractfile(member).read())
            for member in source
            if member.isfile()
        ]
    blob_index = next(
        index for index, (name, _) in enumerate(entries) if name.startswith("blobs/")
    )
    name, _ = entries[blob_index]
    entries[blob_index] = (name, b"changed")

    with pytest.raises(scanner.ImageArchiveError, match="integrity"):
        scanner.scan_image_archive(
            _tar(entries),
            forbidden=b"high-entropy-canary",
            history=b"safe",
        )


@pytest.mark.parametrize("location", ["config", "layer", "history"])
def test_scanner_finds_a_canary_in_each_decoded_image_surface(location: str) -> None:
    scanner = _load_scanner()
    canary = b"high-entropy-canary"
    config = b'{"history":[]}'
    layer = gzip.compress(_tar([("payload.txt", b"safe")]))
    history = b'{"CreatedBy":"safe"}\n'
    if location == "config":
        config = b'{"history":["high-entropy-canary"]}'
    elif location == "layer":
        layer = gzip.compress(_tar([("payload.txt", canary)]))
    else:
        history = b'{"CreatedBy":"high-entropy-canary"}\n'

    with pytest.raises(scanner.ImageArchiveError, match="forbidden bytes") as captured:
        scanner.scan_image_archive(
            _docker_save_archive(layer_payloads=[layer], config=config),
            forbidden=canary,
            history=history,
        )
    assert canary.decode() not in str(captured.value)


def test_scanner_cli_reads_the_canary_from_the_environment_without_disclosure(
    tmp_path: Path,
) -> None:
    canary = "high-entropy-canary"
    archive = tmp_path / "image.tar"
    history = tmp_path / "history.jsonl"
    archive.write_bytes(
        _docker_save_archive(layer_payloads=[_tar([("payload.txt", canary.encode())])])
    )
    history.write_bytes(b"safe")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCANNER),
            "--archive",
            str(archive),
            "--history",
            str(history),
        ],
        env={**os.environ, "AGENTSEEK_IMAGE_SCAN_SENTINEL": canary},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr == "image archive boundary verification failed\n"
    assert canary not in " ".join(completed.args)
    assert canary not in completed.stderr


@pytest.mark.parametrize(
    "archive",
    [
        b"not a tar archive",
        _docker_save_archive(
            layer_payloads=[_tar([("safe", b"safe")])],
            config_reference="missing.json",
        ),
        _docker_save_archive(
            layer_payloads=[_tar([("safe", b"safe")])],
            config_reference="../config.json",
        ),
        _docker_save_archive(
            layer_payloads=[_tar([("safe", b"safe")])],
            extra_entries=[("config.json", b"duplicate")],
        ),
    ],
)
def test_scanner_fails_closed_on_malformed_missing_duplicate_or_escape(
    archive: bytes,
) -> None:
    scanner = _load_scanner()

    with pytest.raises(scanner.ImageArchiveError):
        scanner.scan_image_archive(
            archive,
            forbidden=hashlib.sha256(b"absent").hexdigest().encode(),
            history=b"safe",
        )


def test_scanner_rejects_layer_member_path_escape() -> None:
    scanner = _load_scanner()
    layer = _tar([("../escape", b"safe")])

    with pytest.raises(scanner.ImageArchiveError, match="path"):
        scanner.scan_image_archive(
            _docker_save_archive(layer_payloads=[layer]),
            forbidden=b"absent-canary",
            history=b"safe",
        )
