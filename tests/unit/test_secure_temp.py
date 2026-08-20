from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest

import agentseek_api.secure_temp as secure_temp
from agentseek_api.secure_temp import (
    SecureArtifactError,
    private_artifact,
    private_directory,
    sweep_expired_artifacts,
)

POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="requires POSIX metadata")


@POSIX_ONLY
def test_private_artifact_is_user_only_and_removed_after_failure(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="subprocess failed"):
        with private_artifact(
            tmp_root=tmp_path,
            prefix="agentseek-compose-",
            contents=b"TOKEN='sentinel'\n",
        ) as path:
            assert path.read_bytes() == b"TOKEN='sentinel'\n"
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert path.stat().st_uid == os.getuid()
            raise RuntimeError("subprocess failed")

    assert list(tmp_path.iterdir()) == []


@POSIX_ONLY
def test_private_directory_is_user_only_and_removed_with_contents(
    tmp_path: Path,
) -> None:
    with private_directory(tmp_root=tmp_path, prefix="agentseek-build-") as path:
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
        assert path.stat().st_uid == os.getuid()
        (path / "inventory.json").write_text("{}", encoding="utf-8")

    assert list(tmp_path.iterdir()) == []


@POSIX_ONLY
def test_private_artifact_fails_closed_when_mode_cannot_be_proved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_mode(_fd: int, _path: Path) -> os.stat_result:
        raise SecureArtifactError("Could not prove exclusive access.")

    monkeypatch.setattr(secure_temp, "_verify_open_posix", reject_mode)

    with pytest.raises(SecureArtifactError, match="exclusive access"):
        with private_artifact(
            tmp_root=tmp_path,
            prefix="agentseek-compose-",
            contents=b"private",
        ):
            pytest.fail("an unverified path must never be exposed")

    assert list(tmp_path.iterdir()) == []


@POSIX_ONLY
def test_private_artifact_prep_failure_never_deletes_regular_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = tmp_path / "captured-original"

    def substitute_then_reject(_fd: int, path: Path) -> os.stat_result:
        path.rename(captured)
        path.write_bytes(b"replacement-must-survive")
        path.chmod(0o600)
        raise SecureArtifactError("Could not prove exclusive access.")

    monkeypatch.setattr(secure_temp, "_verify_open_posix", substitute_then_reject)

    with pytest.raises(SecureArtifactError, match="exclusive access"):
        with private_artifact(
            tmp_root=tmp_path,
            prefix="agentseek-compose-",
            contents=b"created-object",
        ):
            pytest.fail("a substituted path must never be exposed")

    assert captured.read_bytes() == b"created-object"
    assert any(
        path.is_file() and path.read_bytes() == b"replacement-must-survive"
        for path in tmp_path.iterdir()
    )


@POSIX_ONLY
@pytest.mark.parametrize("replacement_kind", ["regular", "symlink"])
def test_quarantine_file_never_deletes_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    candidate = tmp_path / "agentseek-compose-race"
    candidate.write_bytes(b"created-object")
    candidate.chmod(0o600)
    expected = candidate.lstat()
    captured = tmp_path / "captured-original"
    symlink_target = tmp_path / "replacement-target"
    if replacement_kind == "symlink":
        symlink_target.write_bytes(b"replacement-must-survive")

    def substitute_then_move(path: Path) -> Path:
        path.rename(captured)
        if replacement_kind == "regular":
            path.write_bytes(b"replacement-must-survive")
            path.chmod(0o600)
        else:
            path.symlink_to(symlink_target)
        quarantine = tmp_path / ".agentseek-quarantine-test"
        path.rename(quarantine)
        return quarantine

    monkeypatch.setattr(
        secure_temp, "_move_to_quarantine", substitute_then_move, raising=False
    )

    removed = secure_temp._quarantine_then_unlink(candidate, expected)

    assert removed is False
    assert captured.read_bytes() == b"created-object"
    if replacement_kind == "regular":
        assert any(
            path.is_file() and path.read_bytes() == b"replacement-must-survive"
            for path in tmp_path.iterdir()
        )
    else:
        assert any(path.is_symlink() for path in tmp_path.iterdir())
        assert symlink_target.read_bytes() == b"replacement-must-survive"


@POSIX_ONLY
def test_quarantine_directory_never_deletes_regular_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "agentseek-build-race"
    candidate.mkdir(mode=0o700)
    (candidate / "owned.txt").write_text("created-object", encoding="utf-8")
    expected = candidate.lstat()
    captured = tmp_path / "captured-original"

    def substitute_then_move(path: Path) -> Path:
        path.rename(captured)
        path.write_bytes(b"replacement-must-survive")
        quarantine = tmp_path / ".agentseek-quarantine-directory-test"
        path.rename(quarantine)
        return quarantine

    monkeypatch.setattr(
        secure_temp, "_move_to_quarantine", substitute_then_move, raising=False
    )

    removed = secure_temp._quarantine_then_rmtree(candidate, expected)

    assert removed is False
    assert (captured / "owned.txt").read_text(encoding="utf-8") == "created-object"
    assert any(
        path.is_file() and path.read_bytes() == b"replacement-must-survive"
        for path in tmp_path.iterdir()
    )


@POSIX_ONLY
def test_private_directory_never_chmods_a_pathname(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_path_chmod(_path: Path, _mode: int) -> None:
        raise AssertionError("private directories must be secured through their handle")

    monkeypatch.setattr(Path, "chmod", reject_path_chmod)

    with private_directory(tmp_root=tmp_path, prefix="agentseek-build-") as path:
        assert path.is_dir()

    assert list(tmp_path.iterdir()) == []


@POSIX_ONLY
def test_private_directory_cleans_captured_object_after_verification_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_directory(_fd: int, _path: Path, _expected: os.stat_result) -> None:
        raise SecureArtifactError("Could not prove exclusive directory access.")

    monkeypatch.setattr(secure_temp, "_verify_open_directory_posix", reject_directory)

    with pytest.raises(SecureArtifactError, match="directory access"):
        with private_directory(tmp_root=tmp_path, prefix="agentseek-build-"):
            pytest.fail("an unverified directory must never be exposed")

    assert list(tmp_path.iterdir()) == []


@POSIX_ONLY
def test_private_artifact_fails_closed_when_owner_cannot_be_proved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(secure_temp, "_current_uid", lambda: os.getuid() + 1)

    with pytest.raises(SecureArtifactError, match="exclusive access"):
        with private_artifact(
            tmp_root=tmp_path,
            prefix="agentseek-compose-",
            contents=b"private",
        ):
            pytest.fail("an unverified path must never be exposed")

    quarantined = list(tmp_path.iterdir())
    assert len(quarantined) == 1
    assert quarantined[0].name.startswith(".agentseek-quarantine-")
    assert quarantined[0].read_bytes() == b"private"


@POSIX_ONLY
def test_private_artifact_rejects_symlink_temporary_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(SecureArtifactError, match="temporary root"):
        with private_artifact(
            tmp_root=linked_root,
            prefix="agentseek-compose-",
            contents=b"private",
        ):
            pytest.fail("a symlink root must never be used")


@POSIX_ONLY
def test_private_artifact_fails_closed_on_symlink_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replacement_target = tmp_path / "replacement-target"
    replacement_target.write_bytes(b"must-not-read")
    original_verify = secure_temp._verify_closed_posix

    def substitute(path: Path, expected: os.stat_result) -> None:
        path.unlink()
        path.symlink_to(replacement_target)
        original_verify(path, expected)

    monkeypatch.setattr(secure_temp, "_verify_closed_posix", substitute)

    with pytest.raises(SecureArtifactError, match="exclusive access"):
        with private_artifact(
            tmp_root=tmp_path,
            prefix="agentseek-compose-",
            contents=b"private",
        ):
            pytest.fail("a substituted path must never be exposed")

    links = [path for path in tmp_path.iterdir() if path.is_symlink()]
    assert len(links) == 1
    assert replacement_target.read_bytes() == b"must-not-read"


@POSIX_ONLY
def test_sweep_removes_only_owned_private_old_regular_artifacts(
    tmp_path: Path,
) -> None:
    old_private = tmp_path / "agentseek-compose-old"
    old_private.write_bytes(b"old")
    old_private.chmod(0o600)
    old_public = tmp_path / "agentseek-compose-public"
    old_public.write_bytes(b"public")
    old_public.chmod(0o644)
    recent = tmp_path / "agentseek-compose-recent"
    recent.write_bytes(b"recent")
    recent.chmod(0o600)
    other_product = tmp_path / "agentseek-build-old"
    other_product.write_bytes(b"other-product")
    other_product.chmod(0o600)
    symlink = tmp_path / "agentseek-compose-link"
    symlink.symlink_to(old_private)
    old = time.time() - 48 * 60 * 60
    os.utime(old_private, (old, old))
    os.utime(old_public, (old, old))
    os.utime(other_product, (old, old))
    os.utime(symlink, (old, old), follow_symlinks=False)

    removed = sweep_expired_artifacts(
        tmp_root=tmp_path,
        prefix="agentseek-compose-",
        older_than_seconds=24 * 60 * 60,
        now=time.time(),
    )

    assert removed == (old_private,)
    assert not old_private.exists()
    assert old_public.exists()
    assert recent.exists()
    assert other_product.exists()
    assert symlink.is_symlink()


@POSIX_ONLY
def test_stale_sweep_never_deletes_regular_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "agentseek-compose-old"
    candidate.write_bytes(b"created-object")
    candidate.chmod(0o600)
    old = time.time() - 48 * 60 * 60
    os.utime(candidate, (old, old))
    captured = tmp_path / "captured-original"
    moved = False

    def substitute_then_move(path: Path) -> Path:
        nonlocal moved
        moved = True
        path.rename(captured)
        path.write_bytes(b"replacement-must-survive")
        path.chmod(0o600)
        quarantine = tmp_path / ".agentseek-quarantine-stale-test"
        path.rename(quarantine)
        return quarantine

    monkeypatch.setattr(
        secure_temp, "_move_to_quarantine", substitute_then_move, raising=False
    )

    removed = sweep_expired_artifacts(
        tmp_root=tmp_path,
        prefix="agentseek-compose-",
        older_than_seconds=24 * 60 * 60,
        now=time.time(),
    )

    assert moved is True
    assert removed == ()
    assert captured.read_bytes() == b"created-object"
    assert any(
        path.is_file() and path.read_bytes() == b"replacement-must-survive"
        for path in tmp_path.iterdir()
    )


@pytest.mark.parametrize(
    ("control", "entries"),
    [
        (
            0x0004 | 0x0400,
            (
                (0, 0x10, 0x001F01FF, "user"),
                (0, 0x10, 0x001F01FF, "system"),
            ),
        ),
        (
            0x0004 | 0x0100 | 0x0400,
            (
                (0, 0x01 | 0x02 | 0x10, 0x001F01FF, "user"),
                (0, 0x01 | 0x02 | 0x10, 0x001F01FF, "system"),
            ),
        ),
    ],
    ids=["inherited-file", "auto-inherited-directory"],
)
def test_windows_descendant_dacl_accepts_effective_inherited_user_and_system_aces(
    control: int,
    entries: tuple[tuple[int, int, int, str], ...],
) -> None:
    secure_temp._validate_windows_descendant_dacl(
        control=control,
        entries=entries,
    )


@pytest.mark.parametrize(
    ("control", "entries"),
    [
        (
            0x0400,
            (
                (0, 0x10, 0x001F01FF, "user"),
                (0, 0x10, 0x001F01FF, "system"),
            ),
        ),
        (
            0x0004 | 0x0008 | 0x0400,
            (
                (0, 0x10, 0x001F01FF, "user"),
                (0, 0x10, 0x001F01FF, "system"),
            ),
        ),
        (
            0x0004 | 0x0400,
            (
                (0, 0x10, 0x001F01FF, "user"),
                (0, 0x10, 0x001F01FF, "other"),
            ),
        ),
        (
            0x0004 | 0x0400,
            (
                (0, 0x10, 0x001F01FF, "user"),
                (1, 0x10, 0x001F01FF, "system"),
            ),
        ),
        (
            0x0004 | 0x0400,
            (
                (0, 0x10, 0x001F01FF, "user"),
                (0, 0x10 | 0x08, 0x001F01FF, "system"),
            ),
        ),
        (
            0x0004 | 0x0400,
            (
                (0, 0x10, 0x001F01FF, "user"),
                (0, 0x10 | 0x04, 0x001F01FF, "system"),
            ),
        ),
        (
            0x0004 | 0x0400,
            (
                (0, 0x10, 0x001F01FF, "user"),
                (0, 0x10, 0x00120089, "system"),
            ),
        ),
        (
            0x0004 | 0x0400,
            (
                (0, 0x10, 0x001F01FF, "user"),
                (0, 0x10, 0x001F01FF, "system"),
                (0, 0x10, 0x001F01FF, "other"),
            ),
        ),
    ],
    ids=[
        "missing-dacl",
        "defaulted-dacl",
        "unexpected-sid",
        "deny-ace",
        "inherit-only-ace",
        "unexpected-inheritance-flag",
        "partial-control-mask",
        "extra-ace",
    ],
)
def test_windows_descendant_dacl_rejects_unsafe_or_ineffective_aces(
    control: int,
    entries: tuple[tuple[int, int, int, str], ...],
) -> None:
    with pytest.raises(SecureArtifactError, match="exclusive Windows access"):
        secure_temp._validate_windows_descendant_dacl(
            control=control,
            entries=entries,
        )


@pytest.mark.parametrize(
    ("directory", "ace_flags"),
    [(False, 0x00), (True, 0x01 | 0x02)],
    ids=["file", "directory"],
)
def test_windows_private_dacl_keeps_strict_explicit_root_contract(
    directory: bool,
    ace_flags: int,
) -> None:
    secure_temp._validate_windows_private_dacl(
        control=0x0004 | 0x1000,
        entries=(
            (0, ace_flags, 0x001F01FF, "user"),
            (0, ace_flags, 0x001F01FF, "system"),
        ),
        directory=directory,
    )

    for control in (0x0004, 0x0004 | 0x1000 | 0x0400):
        with pytest.raises(SecureArtifactError, match="exclusive Windows access"):
            secure_temp._validate_windows_private_dacl(
                control=control,
                entries=(
                    (0, ace_flags, 0x001F01FF, "user"),
                    (0, ace_flags, 0x001F01FF, "system"),
                ),
                directory=directory,
            )


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows security APIs")
def test_windows_private_artifact_dacl_round_trips_by_security_api(
    tmp_path: Path,
) -> None:
    with private_artifact(
        tmp_root=tmp_path,
        prefix="agentseek-compose-",
        contents=b"private",
    ) as path:
        secure_temp._verify_private_dacl(path, directory=False)
        secure_temp._verify_private_dacl(path.parent, directory=True)


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows security APIs")
def test_windows_sweep_removes_private_directory_with_inherited_descendants(
    tmp_path: Path,
) -> None:
    manager = private_directory(tmp_root=tmp_path, prefix="agentseek-build-")
    path = manager.__enter__()
    try:
        nested = path / "nested"
        nested.mkdir()
        (nested / "ordinary.txt").write_text("ordinary", encoding="utf-8")
        now = time.time()
        old = now - 48 * 60 * 60
        os.utime(path, (old, old))

        removed = sweep_expired_artifacts(
            tmp_root=tmp_path,
            prefix="agentseek-build-",
            older_than_seconds=24 * 60 * 60,
            now=now,
        )

        assert removed == (path,)
        assert not path.exists()
    finally:
        manager.__exit__(None, None, None)


@POSIX_ONLY
def test_verify_private_directory_rechecks_owner_mode_and_identity(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "bundle-output"
    directory.mkdir(mode=0o700)

    expected = secure_temp.verify_private_directory(directory)
    assert (expected.st_dev, expected.st_ino) == (
        directory.lstat().st_dev,
        directory.lstat().st_ino,
    )

    directory.chmod(0o755)
    with pytest.raises(SecureArtifactError, match="exclusive directory access"):
        secure_temp.verify_private_directory(directory, expected=expected)


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows security APIs")
def test_verify_private_directory_uses_native_windows_dacl_collection(
    tmp_path: Path,
) -> None:
    with private_directory(tmp_root=tmp_path, prefix="agentseek-build-") as path:
        expected = secure_temp.verify_private_directory(path)
        secure_temp.verify_private_directory(path, expected=expected)
        secure_temp._verify_private_dacl(path, directory=True)


@POSIX_ONLY
def test_create_private_directory_establishes_exact_private_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "persistent-bundle"

    expected = secure_temp.create_private_directory(output)

    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert output.stat().st_uid == os.getuid()
    secure_temp.verify_private_directory(output, expected=expected)

    with pytest.raises(FileExistsError):
        secure_temp.create_private_directory(output)


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows security APIs")
def test_create_private_directory_establishes_native_windows_output_dacl(
    tmp_path: Path,
) -> None:
    output = tmp_path / "persistent-bundle"

    expected = secure_temp.create_private_directory(output)

    secure_temp.verify_private_directory(output, expected=expected)
    secure_temp._verify_private_dacl(output, directory=True)
