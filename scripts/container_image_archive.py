#!/usr/bin/env python3
"""Fail-closed structural scanning for Docker-save and OCI image archives."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import posixpath
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ImageArchiveError(RuntimeError):
    """A value-free image archive validation failure."""


_OCI_CONFIG_MEDIA_TYPES = {
    "application/vnd.oci.image.config.v1+json",
    "application/vnd.docker.container.image.v1+json",
}
_OCI_MANIFEST_MEDIA_TYPES = {
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
}
_OCI_LAYER_ENCODINGS = {
    "application/vnd.oci.image.layer.v1.tar": "tar",
    "application/vnd.oci.image.layer.v1.tar+gzip": "gzip",
    "application/vnd.oci.image.layer.v1.tar+zstd": "zstd",
    "application/vnd.docker.image.rootfs.diff.tar": "tar",
    "application/vnd.docker.image.rootfs.diff.tar.gzip": "gzip",
}


def _safe_name(name: str) -> bool:
    return (
        bool(name)
        and not name.startswith("/")
        and posixpath.normpath(name) == name
        and not any(part in {"", ".", ".."} for part in name.split("/"))
    )


def _canonical_tar_name(name: str) -> str | None:
    if not name or "\x00" in name or name.startswith("/"):
        return None
    trimmed = name.rstrip("/")
    if any(part == ".." for part in trimmed.split("/")):
        return None
    normalized = posixpath.normpath(trimmed)
    if normalized == ".." or normalized.startswith("../"):
        return None
    return normalized


def _json_object(payload: bytes, boundary: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ImageArchiveError(f"{boundary} JSON was malformed") from exc
    if not isinstance(value, dict):
        raise ImageArchiveError(f"{boundary} JSON shape was invalid")
    return value


def _tar_files(payload: bytes, boundary: str) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    names: set[str] = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            for member in archive:
                name = _canonical_tar_name(member.name)
                if name is None or (name == "." and not member.isdir()):
                    raise ImageArchiveError(f"{boundary} member path was unsafe")
                if name == ".":
                    continue
                if name in names:
                    raise ImageArchiveError(f"{boundary} contained duplicate members")
                names.add(name)
                if member.isfile():
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ImageArchiveError(f"{boundary} member was unreadable")
                    files[name] = stream.read()
    except ImageArchiveError:
        raise
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ImageArchiveError(f"{boundary} was malformed") from exc
    return files


def _require_file(
    files: Mapping[str, bytes], reference: object, boundary: str
) -> bytes:
    if not isinstance(reference, str) or not _safe_name(reference):
        raise ImageArchiveError(f"{boundary} reference path was unsafe")
    if reference not in files:
        raise ImageArchiveError(f"{boundary} reference was missing")
    return files[reference]


def _scan_forbidden(payload: bytes, forbidden: bytes) -> None:
    if not forbidden:
        raise ImageArchiveError("forbidden byte sequence was empty")
    if forbidden in payload:
        raise ImageArchiveError("image surface contained forbidden bytes")


def _decode_layer(payload: bytes, encoding: str | None = None) -> bytes:
    if encoding is None:
        if payload.startswith(b"\x1f\x8b"):
            encoding = "gzip"
        elif payload.startswith(b"\x28\xb5\x2f\xfd"):
            encoding = "zstd"
        else:
            encoding = "tar"
    if encoding == "gzip":
        try:
            return gzip.decompress(payload)
        except (OSError, EOFError) as exc:
            raise ImageArchiveError("image layer compression was malformed") from exc
    if encoding == "zstd":
        try:
            import zstandard
        except ImportError as exc:
            raise ImageArchiveError("zstd layer support was unavailable") from exc
        try:
            return zstandard.ZstdDecompressor().decompress(payload)
        except zstandard.ZstdError as exc:
            raise ImageArchiveError("image layer compression was malformed") from exc
    if encoding == "tar":
        return payload
    raise ImageArchiveError("image layer compression was unsupported")


def _scan_layer(
    payload: bytes, forbidden: bytes, *, encoding: str | None = None
) -> None:
    decoded = _decode_layer(payload, encoding)
    _scan_forbidden(decoded, forbidden)
    for member_payload in _tar_files(decoded, "image layer").values():
        _scan_forbidden(member_payload, forbidden)


def _scan_docker_save(files: Mapping[str, bytes], forbidden: bytes) -> None:
    try:
        manifest = json.loads(
            _require_file(files, "manifest.json", "Docker save manifest")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ImageArchiveError("Docker save manifest JSON was malformed") from exc
    if (
        not isinstance(manifest, list)
        or len(manifest) != 1
        or not isinstance(manifest[0], dict)
    ):
        raise ImageArchiveError("Docker save manifest shape was unsupported")
    entry = manifest[0]
    config_reference = entry.get("Config")
    layer_references = entry.get("Layers")
    if not isinstance(layer_references, list) or not layer_references:
        raise ImageArchiveError("Docker save layer references were invalid")
    if (
        not isinstance(config_reference, str)
        or any(not isinstance(reference, str) for reference in layer_references)
        or config_reference in layer_references
    ):
        raise ImageArchiveError("Docker save references were invalid")
    config = _require_file(files, config_reference, "Docker save config")
    _json_object(config, "Docker save config")
    _scan_forbidden(config, forbidden)
    for reference in dict.fromkeys(layer_references):
        _scan_layer(_require_file(files, reference, "Docker save layer"), forbidden)


def _oci_blob(
    files: Mapping[str, bytes],
    descriptor: object,
    media_types: set[str],
    boundary: str,
) -> tuple[bytes, str]:
    if not isinstance(descriptor, dict):
        raise ImageArchiveError(f"{boundary} descriptor shape was invalid")
    digest = descriptor.get("digest")
    size = descriptor.get("size")
    media_type = descriptor.get("mediaType")
    if (
        not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or len(digest) != 71
    ):
        raise ImageArchiveError(f"{boundary} digest was invalid")
    hexadecimal = digest.removeprefix("sha256:")
    if any(character not in "0123456789abcdef" for character in hexadecimal):
        raise ImageArchiveError(f"{boundary} digest was invalid")
    if not isinstance(size, int) or size < 0 or media_type not in media_types:
        raise ImageArchiveError(f"{boundary} descriptor was unsupported")
    payload = _require_file(files, f"blobs/sha256/{hexadecimal}", boundary)
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != hexadecimal:
        raise ImageArchiveError(f"{boundary} blob integrity failed")
    return payload, media_type


def _scan_oci(files: Mapping[str, bytes], forbidden: bytes) -> None:
    layout = _json_object(
        _require_file(files, "oci-layout", "OCI layout"), "OCI layout"
    )
    if layout.get("imageLayoutVersion") != "1.0.0":
        raise ImageArchiveError("OCI layout version was unsupported")
    index = _json_object(_require_file(files, "index.json", "OCI index"), "OCI index")
    manifests = index.get("manifests")
    if not isinstance(manifests, list) or len(manifests) != 1:
        raise ImageArchiveError("OCI index manifest selection was unsupported")
    manifest_bytes, _ = _oci_blob(
        files, manifests[0], _OCI_MANIFEST_MEDIA_TYPES, "OCI manifest"
    )
    manifest = _json_object(manifest_bytes, "OCI manifest")
    config_descriptor = manifest.get("config")
    config_bytes, _ = _oci_blob(
        files, config_descriptor, _OCI_CONFIG_MEDIA_TYPES, "OCI config"
    )
    _json_object(config_bytes, "OCI config")
    _scan_forbidden(config_bytes, forbidden)
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ImageArchiveError("OCI layer descriptors were invalid")
    config_digest = (
        config_descriptor.get("digest") if isinstance(config_descriptor, dict) else None
    )
    descriptors: dict[str, Mapping[str, Any]] = {}
    for descriptor in layers:
        if not isinstance(descriptor, dict):
            raise ImageArchiveError("OCI layer references were invalid")
        digest = descriptor.get("digest")
        if not isinstance(digest, str) or digest == config_digest:
            raise ImageArchiveError("OCI layer references were invalid")
        previous = descriptors.get(digest)
        if previous is not None:
            if descriptor != previous:
                raise ImageArchiveError("OCI layer references conflicted")
            continue
        descriptors[digest] = descriptor
        layer_bytes, media_type = _oci_blob(
            files, descriptor, set(_OCI_LAYER_ENCODINGS), "OCI layer"
        )
        _scan_layer(layer_bytes, forbidden, encoding=_OCI_LAYER_ENCODINGS[media_type])


def scan_image_archive(
    archive: bytes | Path, *, forbidden: bytes, history: bytes
) -> None:
    """Validate and scan every referenced image surface without exposing values."""

    try:
        payload = archive.read_bytes() if isinstance(archive, Path) else bytes(archive)
    except OSError as exc:
        raise ImageArchiveError("image archive could not be read") from exc
    files = _tar_files(payload, "image archive")
    _scan_forbidden(history, forbidden)
    recognized = False
    if "manifest.json" in files:
        _scan_docker_save(files, forbidden)
        recognized = True
    if "oci-layout" in files or "index.json" in files:
        _scan_oci(files, forbidden)
        recognized = True
    if not recognized:
        raise ImageArchiveError("image archive format was unsupported")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    args = parser.parse_args(argv)
    sentinel = os.environ.get("AGENTSEEK_IMAGE_SCAN_SENTINEL")
    if not sentinel:
        raise SystemExit("image archive sentinel was unavailable")
    try:
        scan_image_archive(
            args.archive,
            forbidden=sentinel.encode(),
            history=args.history.read_bytes(),
        )
    except (ImageArchiveError, OSError):
        raise SystemExit("image archive boundary verification failed") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
