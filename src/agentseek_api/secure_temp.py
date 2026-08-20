"""Fail-closed lifecycle helpers for user-private temporary artifacts."""

from __future__ import annotations

import contextlib
import ctypes
import os
import secrets
import shutil
import stat
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class SecureArtifactError(RuntimeError):
    """A value-free failure to prove exclusive access to a temporary object."""


_PRIVATE_FILE_MODE = 0o600
_PRIVATE_DIRECTORY_MODE = 0o700
_MAX_CREATE_ATTEMPTS = 128


def _current_uid() -> int:
    return os.getuid()


def _is_link_or_junction(path: Path, metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _private_root(tmp_root: Path | None) -> Path:
    root = Path(tempfile.gettempdir()) if tmp_root is None else Path(tmp_root)
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise SecureArtifactError(
            "Could not verify the private temporary root."
        ) from exc
    if _is_link_or_junction(root, metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise SecureArtifactError("Could not verify the private temporary root.")
    return root


def _validate_prefix(prefix: str) -> None:
    if (
        not prefix
        or prefix in {".", ".."}
        or Path(prefix).name != prefix
        or "\x00" in prefix
    ):
        raise SecureArtifactError("Temporary artifact prefix is invalid.")


def _candidate_path(root: Path, prefix: str) -> Path:
    return root / f"{prefix}{secrets.token_hex(16)}"


def _same_object(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _verify_open_posix(fd: int, path: Path) -> os.stat_result:
    try:
        opened = os.fstat(fd)
        named = path.lstat()
    except OSError as exc:
        raise SecureArtifactError("Could not prove exclusive access.") from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or not _same_object(opened, named)
        or opened.st_uid != _current_uid()
        or stat.S_IMODE(opened.st_mode) != _PRIVATE_FILE_MODE
        or opened.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
    ):
        raise SecureArtifactError("Could not prove exclusive access.")
    return opened


def _verify_closed_posix(path: Path, expected: os.stat_result) -> None:
    try:
        named = path.lstat()
    except OSError as exc:
        raise SecureArtifactError("Could not prove exclusive access.") from exc
    if (
        not stat.S_ISREG(named.st_mode)
        or not _same_object(named, expected)
        or named.st_uid != _current_uid()
        or stat.S_IMODE(named.st_mode) != _PRIVATE_FILE_MODE
        or named.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
    ):
        raise SecureArtifactError("Could not prove exclusive access.")


def _verify_directory_posix(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SecureArtifactError(
            "Could not prove exclusive directory access."
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != _current_uid()
        or stat.S_IMODE(metadata.st_mode) != _PRIVATE_DIRECTORY_MODE
        or metadata.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
    ):
        raise SecureArtifactError("Could not prove exclusive directory access.")
    return metadata


def _verify_open_windows(  # pragma: no cover - native Windows only
    fd: int, path: Path
) -> os.stat_result:
    try:
        opened = os.fstat(fd)
        named = path.lstat()
    except OSError as exc:
        raise SecureArtifactError("Could not prove exclusive Windows access.") from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or not _same_object(opened, named)
    ):
        raise SecureArtifactError("Could not prove exclusive Windows access.")
    _verify_private_dacl(path)
    return opened


def _verify_closed_windows(  # pragma: no cover - native Windows only
    path: Path, expected: os.stat_result
) -> None:
    try:
        named = path.lstat()
    except OSError as exc:
        raise SecureArtifactError("Could not prove exclusive Windows access.") from exc
    if not stat.S_ISREG(named.st_mode) or not _same_object(named, expected):
        raise SecureArtifactError("Could not prove exclusive Windows access.")
    _verify_private_dacl(path)


def _win32_libraries(  # pragma: no cover - native Windows only
) -> tuple[ctypes.WinDLL, ctypes.WinDLL]:  # type: ignore[name-defined]
    if os.name != "nt":
        raise SecureArtifactError("Windows security APIs are unavailable.")
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    kernel32.CreateDirectoryW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p]
    kernel32.CreateDirectoryW.restype = wintypes.BOOL
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.GetLengthSid.argtypes = [ctypes.c_void_p]
    advapi32.GetLengthSid.restype = wintypes.DWORD
    advapi32.CopySid.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p]
    advapi32.CopySid.restype = wintypes.BOOL
    advapi32.CreateWellKnownSid.argtypes = [
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.CreateWellKnownSid.restype = wintypes.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.ULONG),
    ]
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = (
        wintypes.BOOL
    )
    advapi32.SetFileSecurityW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    advapi32.SetFileSecurityW.restype = wintypes.BOOL
    advapi32.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    advapi32.EqualSid.restype = wintypes.BOOL
    advapi32.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi32.GetSecurityDescriptorControl.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.WORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetSecurityDescriptorControl.restype = wintypes.BOOL
    advapi32.GetAclInformation.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    advapi32.GetAclInformation.restype = wintypes.BOOL
    advapi32.GetAce.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetAce.restype = wintypes.BOOL
    return advapi32, kernel32


def _win32_error(  # pragma: no cover - native Windows only
    message: str,
) -> SecureArtifactError:
    return SecureArtifactError(message)


def _current_user_sid() -> bytes:  # pragma: no cover - native Windows only
    """Return a stable copy of the current process token's user SID."""

    from ctypes import wintypes

    advapi32, kernel32 = _win32_libraries()
    token = wintypes.HANDLE()
    token_query = 0x0008
    token_user = 1
    process = kernel32.GetCurrentProcess()
    if not advapi32.OpenProcessToken(process, token_query, ctypes.byref(token)):
        raise _win32_error("Could not verify the Windows user identity.")
    try:
        size = wintypes.DWORD()
        advapi32.GetTokenInformation(token, token_user, None, 0, ctypes.byref(size))
        if size.value == 0:
            raise _win32_error("Could not verify the Windows user identity.")
        buffer = ctypes.create_string_buffer(size.value)
        if not advapi32.GetTokenInformation(
            token, token_user, buffer, size, ctypes.byref(size)
        ):
            raise _win32_error("Could not verify the Windows user identity.")

        class SidAndAttributes(ctypes.Structure):
            _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]

        sid = ctypes.cast(buffer, ctypes.POINTER(SidAndAttributes)).contents.Sid
        length = advapi32.GetLengthSid(sid)
        if length <= 0:
            raise _win32_error("Could not verify the Windows user identity.")
        copied = ctypes.create_string_buffer(length)
        if not advapi32.CopySid(length, copied, sid):
            raise _win32_error("Could not verify the Windows user identity.")
        return bytes(copied.raw)
    finally:
        kernel32.CloseHandle(token)


def _well_known_system_sid() -> bytes:  # pragma: no cover - native Windows only
    from ctypes import wintypes

    advapi32, _ = _win32_libraries()
    size = wintypes.DWORD(68)
    buffer = ctypes.create_string_buffer(size.value)
    if not advapi32.CreateWellKnownSid(22, None, buffer, ctypes.byref(size)):
        raise _win32_error("Could not verify the Windows SYSTEM identity.")
    return bytes(buffer.raw[: size.value])


def _sid_string(sid_bytes: bytes) -> str:  # pragma: no cover - native Windows only
    from ctypes import wintypes

    advapi32, kernel32 = _win32_libraries()
    sid = ctypes.create_string_buffer(sid_bytes)
    pointer = wintypes.LPWSTR()
    if not advapi32.ConvertSidToStringSidW(sid, ctypes.byref(pointer)):
        raise _win32_error("Could not encode the Windows user identity.")
    try:
        return pointer.value
    finally:
        kernel32.LocalFree(ctypes.cast(pointer, ctypes.c_void_p))


@contextmanager
def _private_security_descriptor(  # pragma: no cover - native Windows only
    *, directory: bool
) -> Iterator[ctypes.c_void_p]:
    from ctypes import wintypes

    advapi32, kernel32 = _win32_libraries()
    user = _sid_string(_current_user_sid())
    inheritance = "OICI" if directory else ""
    sddl = f"O:{user}D:P(A;{inheritance};FA;;;{user})(A;{inheritance};FA;;;SY)"
    descriptor = ctypes.c_void_p()
    descriptor_size = wintypes.ULONG()
    if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        sddl, 1, ctypes.byref(descriptor), ctypes.byref(descriptor_size)
    ):
        raise _win32_error("Could not establish exclusive Windows access.")
    try:
        yield descriptor
    finally:
        kernel32.LocalFree(descriptor)


def _apply_private_dacl(  # pragma: no cover - native Windows only
    path: Path, *, directory: bool = False
) -> None:
    """Install a protected owner/DACL containing only user and SYSTEM access."""

    advapi32, _ = _win32_libraries()
    with _private_security_descriptor(directory=directory) as descriptor:
        owner_information = 0x00000001
        dacl_information = 0x00000004
        protected_dacl_information = 0x80000000
        if not advapi32.SetFileSecurityW(
            str(path),
            owner_information | dacl_information | protected_dacl_information,
            descriptor,
        ):
            raise _win32_error("Could not establish exclusive Windows access.")


def _create_private_windows_directory(  # pragma: no cover - native Windows only
    root: Path, prefix: str
) -> tuple[Path, os.stat_result]:
    from ctypes import wintypes

    _, kernel32 = _win32_libraries()

    class SecurityAttributes(ctypes.Structure):
        _fields_ = [
            ("nLength", wintypes.DWORD),
            ("lpSecurityDescriptor", ctypes.c_void_p),
            ("bInheritHandle", wintypes.BOOL),
        ]

    for _ in range(_MAX_CREATE_ATTEMPTS):
        candidate = _candidate_path(root, prefix)
        with _private_security_descriptor(directory=True) as descriptor:
            attributes = SecurityAttributes(
                ctypes.sizeof(SecurityAttributes), descriptor, False
            )
            if kernel32.CreateDirectoryW(str(candidate), ctypes.byref(attributes)):
                expected = candidate.lstat()
                try:
                    _verify_private_dacl(candidate)
                except SecureArtifactError:
                    _safe_remove_directory(candidate, expected)
                    raise
                return candidate, expected
            if ctypes.get_last_error() != 183:
                raise SecureArtifactError("Could not create a private directory.")
    raise SecureArtifactError("Could not create a private directory.")


def _sid_matches(  # pragma: no cover - native Windows only
    left: ctypes.c_void_p, right_bytes: bytes
) -> bool:
    advapi32, _ = _win32_libraries()
    right = ctypes.create_string_buffer(right_bytes)
    return bool(advapi32.EqualSid(left, right))


def _verify_private_dacl(  # pragma: no cover - native Windows only
    path: Path,
) -> None:
    """Read owner and effective ACEs back without localized command output."""

    from ctypes import wintypes

    advapi32, kernel32 = _win32_libraries()
    owner = ctypes.c_void_p()
    dacl = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    owner_information = 0x00000001
    dacl_information = 0x00000004
    result = advapi32.GetNamedSecurityInfoW(
        str(path),
        1,
        owner_information | dacl_information,
        ctypes.byref(owner),
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(descriptor),
    )
    if result != 0 or not owner.value or not dacl.value or not descriptor.value:
        if descriptor.value:
            kernel32.LocalFree(descriptor)
        raise _win32_error("Could not prove exclusive Windows access.")
    try:
        user_sid = _current_user_sid()
        system_sid = _well_known_system_sid()
        if not _sid_matches(owner, user_sid):
            raise _win32_error("Could not prove exclusive Windows access.")

        control = wintypes.WORD()
        revision = wintypes.DWORD()
        if (
            not advapi32.GetSecurityDescriptorControl(
                descriptor, ctypes.byref(control), ctypes.byref(revision)
            )
            or not control.value & 0x1000
        ):
            raise _win32_error("Could not prove exclusive Windows access.")

        class AclSizeInformation(ctypes.Structure):
            _fields_ = [
                ("AceCount", wintypes.DWORD),
                ("AclBytesInUse", wintypes.DWORD),
                ("AclBytesFree", wintypes.DWORD),
            ]

        info = AclSizeInformation()
        if not advapi32.GetAclInformation(
            dacl, ctypes.byref(info), ctypes.sizeof(info), 2
        ):
            raise _win32_error("Could not prove exclusive Windows access.")
        seen_user = False
        seen_system = False
        if info.AceCount != 2:
            raise _win32_error("Could not prove exclusive Windows access.")
        for index in range(info.AceCount):
            ace = ctypes.c_void_p()
            if not advapi32.GetAce(dacl, index, ctypes.byref(ace)) or not ace.value:
                raise _win32_error("Could not prove exclusive Windows access.")
            header = ctypes.string_at(ace, 8)
            ace_type = header[0]
            ace_flags = header[1]
            mask = int.from_bytes(header[4:8], byteorder="little")
            ace_sid = ctypes.c_void_p(ace.value + 8)
            if ace_type != 0 or ace_flags & 0x10 or mask & 0x1F01FF != 0x1F01FF:
                raise _win32_error("Could not prove exclusive Windows access.")
            if _sid_matches(ace_sid, user_sid):
                seen_user = True
            elif _sid_matches(ace_sid, system_sid):
                seen_system = True
            else:
                raise _win32_error("Could not prove exclusive Windows access.")
        if not seen_user or not seen_system:
            raise _win32_error("Could not prove exclusive Windows access.")
    finally:
        kernel32.LocalFree(descriptor)


def _safe_unlink(path: Path, expected: os.stat_result | None = None) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        return
    if _is_link_or_junction(path, metadata):
        return
    if expected is not None and not _same_object(metadata, expected):
        return
    with contextlib.suppress(OSError):
        path.unlink()


def _safe_remove_directory(path: Path, expected: os.stat_result | None) -> None:
    try:
        metadata = path.lstat()
    except OSError:
        return
    if _is_link_or_junction(path, metadata):
        return
    if expected is not None and not _same_object(metadata, expected):
        return
    with contextlib.suppress(OSError):
        shutil.rmtree(path)


@contextmanager
def private_artifact(
    *,
    contents: bytes,
    prefix: str,
    tmp_root: Path | None = None,
) -> Iterator[Path]:
    """Create, verify, expose, and remove one private regular file."""

    root = _private_root(tmp_root)
    _validate_prefix(prefix)
    path: Path | None = None
    private_parent: Path | None = None
    private_parent_expected: os.stat_result | None = None
    fd: int | None = None
    expected: os.stat_result | None = None
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if os.name == "nt":  # pragma: no cover - native Windows only
        private_parent, private_parent_expected = _create_private_windows_directory(
            root, prefix
        )
        creation_root = private_parent
        creation_prefix = "environment-"
    else:
        creation_root = root
        creation_prefix = prefix
    for _ in range(_MAX_CREATE_ATTEMPTS):
        candidate = _candidate_path(creation_root, creation_prefix)
        try:
            fd = os.open(candidate, flags, _PRIVATE_FILE_MODE)
        except FileExistsError:
            continue
        except OSError as exc:
            if private_parent is not None:
                _safe_remove_directory(private_parent, private_parent_expected)
            raise SecureArtifactError("Could not create a private artifact.") from exc
        path = candidate
        break
    if path is None or fd is None:
        if private_parent is not None:
            _safe_remove_directory(private_parent, private_parent_expected)
        raise SecureArtifactError("Could not create a private artifact.")

    try:
        view = memoryview(contents)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise SecureArtifactError("Could not write a private artifact.")
            view = view[written:]
        os.fsync(fd)
        if os.name == "nt":  # pragma: no cover - native Windows only
            _apply_private_dacl(path)
            expected = _verify_open_windows(fd, path)
        else:
            os.fchmod(fd, _PRIVATE_FILE_MODE)
            expected = _verify_open_posix(fd, path)
    except (OSError, SecureArtifactError) as exc:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
            fd = None
        _safe_unlink(path, expected)
        if private_parent is not None:
            _safe_remove_directory(private_parent, private_parent_expected)
        if isinstance(exc, SecureArtifactError):
            raise
        raise SecureArtifactError("Could not prepare a private artifact.") from exc

    try:
        try:
            os.close(fd)
        except OSError as exc:
            raise SecureArtifactError("Could not prepare a private artifact.") from exc
        fd = None
        if os.name == "nt":  # pragma: no cover - native Windows only
            assert expected is not None
            _verify_closed_windows(path, expected)
        else:
            assert expected is not None
            _verify_closed_posix(path, expected)
        yield path
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        _safe_unlink(path, expected)
        if private_parent is not None:
            _safe_remove_directory(private_parent, private_parent_expected)


@contextmanager
def private_directory(*, prefix: str, tmp_root: Path | None = None) -> Iterator[Path]:
    """Create, verify, expose, and recursively remove a private directory."""

    root = _private_root(tmp_root)
    _validate_prefix(prefix)
    path: Path | None = None
    expected: os.stat_result | None = None
    if os.name == "nt":  # pragma: no cover - native Windows only
        path, expected = _create_private_windows_directory(root, prefix)
    else:
        for _ in range(_MAX_CREATE_ATTEMPTS):
            candidate = _candidate_path(root, prefix)
            try:
                os.mkdir(candidate, _PRIVATE_DIRECTORY_MODE)
            except FileExistsError:
                continue
            except OSError as exc:
                raise SecureArtifactError(
                    "Could not create a private directory."
                ) from exc
            path = candidate
            break
    if path is None:
        raise SecureArtifactError("Could not create a private directory.")
    try:
        if os.name == "nt":  # pragma: no cover - native Windows only
            _verify_private_dacl(path)
            assert expected is not None
        else:
            path.chmod(_PRIVATE_DIRECTORY_MODE)
            expected = _verify_directory_posix(path)
    except OSError as exc:
        _safe_remove_directory(path, expected)
        raise SecureArtifactError("Could not prepare a private directory.") from exc
    try:
        yield path
    finally:
        _safe_remove_directory(path, expected)


def sweep_expired_artifacts(
    *,
    prefix: str,
    older_than_seconds: float,
    tmp_root: Path | None = None,
    now: float | None = None,
) -> tuple[Path, ...]:
    """Remove only old, same-user, still-private artifact objects."""

    if older_than_seconds < 0:
        raise SecureArtifactError("Artifact expiry must not be negative.")
    root = _private_root(tmp_root)
    _validate_prefix(prefix)
    cutoff = (time.time() if now is None else now) - older_than_seconds
    removed: list[Path] = []
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        raise SecureArtifactError("Could not inspect stale private artifacts.") from exc
    for entry in entries:
        if not entry.name.startswith(prefix):
            continue
        candidate = root / entry.name
        try:
            metadata = candidate.lstat()
            if (
                _is_link_or_junction(candidate, metadata)
                or metadata.st_mtime >= cutoff
                or candidate.resolve(strict=True).parent != root.resolve(strict=True)
            ):
                continue
            if (  # pragma: no cover - native Windows only
                os.name == "nt" and stat.S_ISDIR(metadata.st_mode)
            ):
                _verify_private_dacl(candidate)
                descendants = tuple(candidate.rglob("*"))
                if any(descendant.is_symlink() for descendant in descendants):
                    continue
                for descendant in descendants:
                    if descendant.is_dir():
                        _verify_private_dacl(descendant)
                    elif descendant.is_file():
                        _verify_private_dacl(descendant)
                    else:
                        raise SecureArtifactError(
                            "Could not prove exclusive Windows access."
                        )
                _safe_remove_directory(candidate, metadata)
                if candidate.exists():
                    continue
            elif not stat.S_ISREG(metadata.st_mode):
                continue
            elif os.name == "nt":  # pragma: no cover - native Windows only
                _verify_private_dacl(candidate)
                candidate.unlink()
            elif (
                metadata.st_uid != _current_uid()
                or stat.S_IMODE(metadata.st_mode) != _PRIVATE_FILE_MODE
                or metadata.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
            ):
                continue
            else:
                candidate.unlink()
        except (OSError, SecureArtifactError):
            continue
        removed.append(candidate)
    return tuple(sorted(removed))


__all__ = [
    "SecureArtifactError",
    "private_artifact",
    "private_directory",
    "sweep_expired_artifacts",
]
