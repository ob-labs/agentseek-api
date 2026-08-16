from __future__ import annotations

import ctypes
import errno
import math
import os
import signal
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from types import FrameType
from typing import Protocol, Self


_IS_WINDOWS = os.name == "nt"
_MANAGED_SIGNALS = (signal.SIGINT, signal.SIGTERM)
_SUPERVISION_ERROR = "Runtime child supervision failed."
_WINDOWS_WAIT_POLL_SECONDS = 0.05
_DARWIN_P_PID = 1
_DARWIN_WNOHANG = 0x00000001
_DARWIN_WEXITED = 0x00000004
_DARWIN_WNOWAIT = 0x00000020
_CLD_EXITED = 1
_CLD_KILLED = 2
_CLD_DUMPED = 3
_DARWIN_WAITID_FUNCTION = None


class ProcessSupervisionError(RuntimeError):
    """A value-free failure at the child-process ownership boundary."""

    def __init__(self, _detail: object | None = None) -> None:
        super().__init__(_SUPERVISION_ERROR)


class _ForwardedSignal(Exception):
    def __init__(self, signum: int) -> None:
        super().__init__()
        self.signum = signum


class _SignalTarget(Protocol):
    def forward_signal(self, signum: int) -> None: ...


class ForwardingSignalGuard:
    """Own temporary foreground handlers without exposing an unguarded spawn gap."""

    def __init__(self) -> None:
        self._state = "new"
        self._child: _SignalTarget | None = None
        self._pending_signal: int | None = None
        self._previous_handlers: dict[int, object] = {}
        self._installed_signals: list[int] = []
        self._original_mask: set[signal.Signals] | None = None
        self._mask_is_blocked = False
        self._cleanup_forward_failed = False
        self._installed_handler = self._handle_signal

    def __enter__(self) -> Self:
        if threading.current_thread() is not threading.main_thread():
            raise ProcessSupervisionError()
        self._state = "acquiring"
        try:
            self._block_for_handler_installation()
            if self._original_mask is not None and any(
                signum in self._original_mask for signum in _MANAGED_SIGNALS
            ):
                raise ProcessSupervisionError()
            for signum in _MANAGED_SIGNALS:
                self._previous_handlers[signum] = signal.getsignal(signum)
            for signum in _MANAGED_SIGNALS:
                signal.signal(signum, self._installed_handler)
                self._installed_signals.append(signum)
            for signum in _MANAGED_SIGNALS:
                if signal.getsignal(signum) is not self._installed_handler:
                    raise ProcessSupervisionError()
            self._restore_entry_mask()
            return self
        except BaseException as exc:
            self._state = "cleanup"
            self._restore_after_failed_entry()
            if isinstance(exc, ProcessSupervisionError):
                raise
            raise ProcessSupervisionError() from exc

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback,
    ) -> bool:
        self.begin_cleanup()
        restore_failed = False
        try:
            self._restore_handlers_and_mask()
        except ProcessSupervisionError:
            restore_failed = True
        self._state = "closed"
        if restore_failed or self._cleanup_forward_failed:
            raise ProcessSupervisionError()
        return False

    def attach(self, child: _SignalTarget) -> None:
        if self._state != "acquiring" or self._child is not None:
            raise ProcessSupervisionError()
        self._child = child
        self._state = "waiting"
        pending_signal = self._pending_signal
        if pending_signal is None:
            return
        self._state = "delivering"
        self._pending_signal = None
        raise _ForwardedSignal(pending_signal)

    def begin_cleanup(self) -> None:
        if self._state != "closed":
            self._state = "cleanup"

    def _handle_signal(
        self,
        signum: int,
        _frame: FrameType | None,
    ) -> None:
        if self._state == "acquiring":
            if self._pending_signal is None:
                self._pending_signal = signum
            return
        if self._state == "waiting":
            self._state = "delivering"
            if self._pending_signal is not None:
                signum = self._pending_signal
                self._pending_signal = None
            raise _ForwardedSignal(signum)
        if self._state in {"delivering", "cleanup"}:
            if self._child is None:
                return
            try:
                self._child.forward_signal(signum)
            except BaseException:
                self._cleanup_forward_failed = True

    def _block_for_handler_installation(self) -> None:
        if _IS_WINDOWS or not hasattr(signal, "pthread_sigmask"):
            return
        self._original_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK,
            set(_MANAGED_SIGNALS),
        )
        self._mask_is_blocked = True

    def _restore_entry_mask(self) -> None:
        if not self._mask_is_blocked or self._original_mask is None:
            return
        signal.pthread_sigmask(signal.SIG_SETMASK, self._original_mask)
        self._mask_is_blocked = False

    def _restore_after_failed_entry(self) -> None:
        failed = False
        for signum in reversed(self._installed_signals):
            previous = self._previous_handlers.get(signum)
            if previous is None:
                continue
            try:
                signal.signal(signum, previous)
            except BaseException:
                failed = True
        try:
            self._restore_entry_mask()
        except BaseException:
            failed = True
        if failed:
            raise ProcessSupervisionError()

    def _restore_handlers_and_mask(self) -> None:
        failed = False
        mask_temporarily_blocked = False
        if not _IS_WINDOWS and hasattr(signal, "pthread_sigmask"):
            try:
                signal.pthread_sigmask(
                    signal.SIG_BLOCK,
                    set(_MANAGED_SIGNALS),
                )
                mask_temporarily_blocked = True
            except BaseException:
                failed = True
        for signum in reversed(self._installed_signals):
            previous = self._previous_handlers[signum]
            try:
                signal.signal(signum, previous)
            except BaseException:
                failed = True
        if mask_temporarily_blocked and self._original_mask is not None:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, self._original_mask)
            except BaseException:
                failed = True
        if failed:
            raise ProcessSupervisionError()


def _decode_waitid_exit(result, *, expected_pid: int) -> int:
    if result is None or result.si_pid != expected_pid:
        raise ProcessSupervisionError()
    if result.si_code == _CLD_EXITED:
        return int(result.si_status)
    if result.si_code in (_CLD_KILLED, _CLD_DUMPED):
        return -int(result.si_status)
    raise ProcessSupervisionError()


class _DarwinSigval(ctypes.Union):
    _fields_ = [
        ("sival_int", ctypes.c_int),
        ("sival_ptr", ctypes.c_void_p),
    ]


class _DarwinSiginfo(ctypes.Structure):
    _fields_ = [
        ("si_signo", ctypes.c_int),
        ("si_errno", ctypes.c_int),
        ("si_code", ctypes.c_int),
        ("si_pid", ctypes.c_int),
        ("si_uid", ctypes.c_uint),
        ("si_status", ctypes.c_int),
        ("si_addr", ctypes.c_void_p),
        ("si_value", _DarwinSigval),
        ("si_band", ctypes.c_long),
        ("reserved", ctypes.c_ulong * 7),
    ]


def _darwin_libc_waitid():
    global _DARWIN_WAITID_FUNCTION
    if _DARWIN_WAITID_FUNCTION is not None:
        return _DARWIN_WAITID_FUNCTION
    if ctypes.sizeof(_DarwinSiginfo) != 104:
        raise ProcessSupervisionError()
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        waitid = libc.waitid
        waitid.argtypes = [
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.POINTER(_DarwinSiginfo),
            ctypes.c_int,
        ]
        waitid.restype = ctypes.c_int
    except BaseException as exc:
        raise ProcessSupervisionError() from exc
    _DARWIN_WAITID_FUNCTION = waitid
    return waitid


def _darwin_waitid_no_reap(pid: int, *, nohang: bool) -> int | None:
    options = _DARWIN_WEXITED | _DARWIN_WNOWAIT
    if nohang:
        options |= _DARWIN_WNOHANG
    while True:
        information = _DarwinSiginfo()
        ctypes.set_errno(0)
        try:
            result = _darwin_libc_waitid()(
                _DARWIN_P_PID,
                pid,
                ctypes.byref(information),
                options,
            )
        except (KeyboardInterrupt, _ForwardedSignal):
            raise
        except BaseException as exc:
            raise ProcessSupervisionError() from exc
        if result == 0:
            if information.si_pid == 0:
                return None
            return _decode_waitid_exit(information, expected_pid=pid)
        if ctypes.get_errno() != errno.EINTR:
            raise ProcessSupervisionError()


def _require_posix_supervision_support() -> None:
    required_names = (
        "P_PID",
        "WEXITED",
        "WNOWAIT",
        "WNOHANG",
        "CLD_EXITED",
        "CLD_KILLED",
        "CLD_DUMPED",
    )
    if sys.platform == "darwin" and not hasattr(os, "waitid"):
        _darwin_libc_waitid()
        return
    if (
        not hasattr(os, "waitid")
        or any(not hasattr(os, name) for name in required_names)
        or not (sys.platform == "darwin" or sys.platform.startswith("linux"))
    ):
        raise ProcessSupervisionError()


def _waitid_no_reap(pid: int, *, nohang: bool) -> int | None:
    if sys.platform == "darwin" and not hasattr(os, "waitid"):
        return _darwin_waitid_no_reap(pid, nohang=nohang)
    _require_posix_supervision_support()
    options = os.WEXITED | os.WNOWAIT
    if nohang:
        options |= os.WNOHANG
    try:
        result = os.waitid(os.P_PID, pid, options)
    except (KeyboardInterrupt, _ForwardedSignal):
        raise
    except BaseException as exc:
        raise ProcessSupervisionError() from exc
    if result is None:
        return None
    return _decode_waitid_exit(result, expected_pid=pid)


def _linux_process_group_members(pgid: int) -> set[int]:
    members: set[int] = set()
    try:
        entries = os.scandir("/proc")
    except BaseException as exc:
        raise ProcessSupervisionError() from exc
    with entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            try:
                with open(
                    f"/proc/{entry.name}/stat",
                    encoding="utf-8",
                ) as stat_file:
                    stat_text = stat_file.read()
            except FileNotFoundError:
                continue
            except BaseException as exc:
                raise ProcessSupervisionError() from exc
            command_end = stat_text.rfind(")")
            fields = stat_text[command_end + 1 :].split()
            if command_end < 0 or len(fields) < 3:
                raise ProcessSupervisionError()
            try:
                observed_pgid = int(fields[2])
                pid = int(entry.name)
            except ValueError as exc:
                raise ProcessSupervisionError() from exc
            if observed_pgid == pgid:
                members.add(pid)
    return members


def _darwin_process_group_members(pgid: int) -> set[int]:
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        list_group_pids = libproc.proc_listpgrppids
        list_group_pids.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        list_group_pids.restype = ctypes.c_int
    except BaseException as exc:
        raise ProcessSupervisionError() from exc

    def list_pids(buffer, size: int) -> int:
        ctypes.set_errno(0)
        count = list_group_pids(pgid, buffer, size)
        call_errno = ctypes.get_errno()
        if count < 0 or (count == 0 and call_errno != 0):
            raise ProcessSupervisionError()
        return count

    try:
        capacity = list_pids(None, 0)
        if capacity == 0:
            return set()
        capacity = max(16, capacity)
        for _attempt in range(3):
            buffer = (ctypes.c_int * capacity)()
            count = list_pids(
                ctypes.cast(buffer, ctypes.c_void_p),
                ctypes.sizeof(buffer),
            )
            if count < capacity:
                return {int(pid) for pid in buffer[:count] if pid > 0}
            capacity *= 2
    except (KeyboardInterrupt, _ForwardedSignal):
        raise
    except ProcessSupervisionError:
        raise
    except BaseException as exc:
        raise ProcessSupervisionError() from exc
    raise ProcessSupervisionError()


def _process_group_has_other_members(pgid: int, leader_pid: int) -> bool:
    if pgid <= 0 or leader_pid <= 0 or pgid != leader_pid:
        raise ProcessSupervisionError()
    if sys.platform == "darwin":
        members = _darwin_process_group_members(pgid)
    elif sys.platform.startswith("linux"):
        members = _linux_process_group_members(pgid)
    else:
        raise ProcessSupervisionError()
    return any(pid != leader_pid for pid in members)


class _PosixChild:
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process
        self._pgid = process.pid
        self._observed_exit_code: int | None = None
        self._direct_reaped = False
        self._group_signal_allowed = True
        self._cleanup_error = False

    @classmethod
    def start(
        cls,
        command: list[str],
        *,
        env: dict[str, str],
        cwd: str | None,
    ) -> Self:
        _require_posix_supervision_support()
        if sys.platform == "darwin":
            _darwin_process_group_members(os.getpgrp())
        else:
            _linux_process_group_members(os.getpgrp())
        try:
            process = subprocess.Popen(
                command,
                env=env,
                cwd=cwd,
                start_new_session=True,
            )
        except BaseException as exc:
            raise ProcessSupervisionError() from exc
        return cls(process)

    def wait(self) -> int:
        try:
            if not self._observe_exit(nohang=False):
                raise ProcessSupervisionError()
            assert self._observed_exit_code is not None
            return self._observed_exit_code
        except (KeyboardInterrupt, _ForwardedSignal):
            raise
        except ProcessSupervisionError:
            raise
        except BaseException as exc:
            raise ProcessSupervisionError() from exc

    def forward_signal(self, signum: int) -> None:
        try:
            self._signal_group(signum)
        except ProcessSupervisionError:
            raise
        except BaseException as exc:
            raise ProcessSupervisionError() from exc

    def forward_and_reap(self, signum: int, *, timeout: float) -> None:
        self._clean_and_reap(
            signum,
            timeout=timeout,
            signal_only_if_members=False,
        )

    def terminate_and_reap(self, *, timeout: float) -> None:
        self._clean_and_reap(
            signal.SIGTERM,
            timeout=timeout,
            signal_only_if_members=False,
        )

    def close_remaining_tree(self, *, timeout: float) -> None:
        if self._observed_exit_code is None and not self._direct_reaped:
            raise ProcessSupervisionError()
        self._clean_and_reap(
            signal.SIGTERM,
            timeout=timeout,
            signal_only_if_members=True,
        )

    def ensure_closed(self, *, timeout: float) -> None:
        if self._direct_reaped:
            if self._cleanup_error:
                raise ProcessSupervisionError()
            return
        if self._observed_exit_code is not None:
            self.close_remaining_tree(timeout=timeout)
            return
        self.terminate_and_reap(timeout=timeout)

    def close(self) -> None:
        return None

    def _validate_process_group(self) -> None:
        if self._pgid <= 0 or self._pgid == os.getpgrp():
            raise ProcessSupervisionError()
        try:
            observed_pgid = os.getpgid(self._process.pid)
        except ProcessLookupError:
            if self._observed_exit_code is not None and not self._direct_reaped:
                return
            raise ProcessSupervisionError() from None
        except BaseException as exc:
            raise ProcessSupervisionError() from exc
        if observed_pgid != self._pgid:
            raise ProcessSupervisionError()

    def _signal_group(self, signum: int) -> None:
        if not self._group_signal_allowed or self._direct_reaped:
            return
        self._validate_process_group()
        try:
            os.killpg(self._pgid, signum)
        except ProcessLookupError:
            return
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return
            raise ProcessSupervisionError() from exc

    def _observe_exit(self, *, nohang: bool) -> bool:
        if self._observed_exit_code is not None:
            return True
        exit_code = _waitid_no_reap(self._process.pid, nohang=nohang)
        if exit_code is None:
            return False
        self._observed_exit_code = exit_code
        return True

    def _has_other_group_members(self) -> bool:
        return _process_group_has_other_members(
            self._pgid,
            self._process.pid,
        )

    def _wait_for_owned_tree_exit(self, *, deadline: float) -> tuple[bool, bool]:
        failure = False
        while True:
            try:
                direct_exited = self._observe_exit(nohang=True)
                other_members = self._has_other_group_members()
            except ProcessSupervisionError:
                direct_exited = False
                other_members = True
                failure = True
            if direct_exited and not other_members:
                return True, failure
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False, failure
            time.sleep(min(0.02, remaining))

    def _reap_observed_child(self) -> None:
        if self._direct_reaped:
            return
        if self._observed_exit_code is None:
            raise ProcessSupervisionError()
        expected_exit_code = self._observed_exit_code
        try:
            observed_exit_code = self._wait_and_mark_direct_reaped()
        except (KeyboardInterrupt, _ForwardedSignal):
            raise
        except BaseException as exc:
            raise ProcessSupervisionError() from exc
        if observed_exit_code != expected_exit_code:
            raise ProcessSupervisionError()

    def _wait_and_mark_direct_reaped(self) -> int:
        previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK,
            set(_MANAGED_SIGNALS),
        )
        self._group_signal_allowed = False
        try:
            observed_exit_code = self._process.wait(timeout=0.0)
            self._direct_reaped = True
            return observed_exit_code
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    def _clean_and_reap(
        self,
        signum: int,
        *,
        timeout: float,
        signal_only_if_members: bool,
    ) -> None:
        if self._direct_reaped:
            if self._cleanup_error:
                raise ProcessSupervisionError()
            return
        failure = False

        if self._observed_exit_code is None:
            try:
                self._observe_exit(nohang=True)
            except ProcessSupervisionError:
                failure = True

        should_signal = True
        if signal_only_if_members:
            try:
                should_signal = self._has_other_group_members()
            except ProcessSupervisionError:
                failure = True

        if not should_signal and self._observed_exit_code is not None:
            try:
                self._reap_observed_child()
            except ProcessSupervisionError:
                failure = True
            if failure:
                self._cleanup_error = True
                raise ProcessSupervisionError()
            return

        try:
            self._signal_group(signum)
        except ProcessSupervisionError:
            failure = True
        soft_deadline = time.monotonic() + timeout
        complete, wait_failed = self._wait_for_owned_tree_exit(
            deadline=soft_deadline,
        )
        failure = failure or wait_failed

        if not complete:
            try:
                self._signal_group(signal.SIGKILL)
            except ProcessSupervisionError:
                failure = True
            hard_deadline = time.monotonic() + timeout
            complete, hard_wait_failed = self._wait_for_owned_tree_exit(
                deadline=hard_deadline,
            )
            failure = failure or hard_wait_failed

        if self._observed_exit_code is not None:
            try:
                self._reap_observed_child()
            except ProcessSupervisionError:
                failure = True
        else:
            try:
                self._wait_and_mark_direct_reaped()
            except subprocess.TimeoutExpired:
                pass
            except (KeyboardInterrupt, _ForwardedSignal):
                raise
            except BaseException:
                failure = True
            else:
                failure = True

        if failure or not complete or not self._direct_reaped:
            if self._direct_reaped:
                self._cleanup_error = True
            raise ProcessSupervisionError()


class _Win32ApiProtocol(Protocol):
    def create_job(self): ...

    def set_kill_on_close(self, job) -> None: ...

    def create_suspended_process(
        self,
        command: list[str],
        *,
        env: dict[str, str],
        cwd: str | None,
        job,
    ): ...

    def resume_thread(self, thread) -> None: ...

    def terminate_job(self, job) -> None: ...

    def wait_process(self, process, timeout: float | None) -> bool: ...

    def process_exit_code(self, process) -> int: ...

    def wait_for_job_empty(self, job, timeout: float) -> bool: ...

    def send_ctrl_break(self, process_id: int) -> None: ...

    def close_handle(self, handle) -> None: ...


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


class _STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(wintypes.BYTE)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [
        ("StartupInfo", _STARTUPINFOW),
        ("lpAttributeList", ctypes.c_void_p),
    ]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class _Win32AttributeList:
    def __init__(self, *, buffer, pointer, handle_array, job_array) -> None:
        self.buffer = buffer
        self.pointer = pointer
        self.handle_array = handle_array
        self.job_array = job_array


class _WindowsLaunchNativeProtocol(Protocol):
    def get_standard_handle(self, stream: int): ...

    def open_null_handle(self, stream: int): ...

    def duplicate_inheritable_handle(self, handle): ...

    def create_attribute_list(self, handles, jobs): ...

    def create_suspended_process(
        self,
        command: list[str],
        *,
        env: dict[str, str],
        cwd: str | None,
        standard_handles,
        attribute_list,
    ): ...

    def delete_handle_list(self, attribute_list) -> None: ...

    def close_handle(self, handle) -> None: ...

    def abort_suspended_process(self, process, thread) -> None: ...


class _WindowsProcessLauncher:
    def __init__(self, native: _WindowsLaunchNativeProtocol) -> None:
        self._native = native

    def create(
        self,
        command: list[str],
        *,
        env: dict[str, str],
        cwd: str | None,
        job,
    ):
        duplicates: list[object] = []
        owned_sources: list[object] = []
        attribute_list = None
        result = None
        failure: BaseException | None = None
        try:
            standard_handles = []
            for stream in (-10, -11, -12):
                handle = self._native.get_standard_handle(stream)
                if handle is None:
                    handle = self._native.open_null_handle(stream)
                    owned_sources.append(handle)
                standard_handles.append(handle)
            for handle in standard_handles:
                duplicates.append(self._native.duplicate_inheritable_handle(handle))
            attribute_list = self._native.create_attribute_list(
                duplicates,
                (job,),
            )
            result = self._native.create_suspended_process(
                command,
                env=env,
                cwd=cwd,
                standard_handles=tuple(duplicates),
                attribute_list=attribute_list,
            )
        except BaseException as exc:
            failure = exc

        try:
            if attribute_list is not None:
                self._native.delete_handle_list(attribute_list)
        except BaseException as exc:
            if failure is None:
                failure = exc
        for handle in duplicates:
            try:
                self._native.close_handle(handle)
            except BaseException as exc:
                if failure is None:
                    failure = exc
        for handle in owned_sources:
            try:
                self._native.close_handle(handle)
            except BaseException as exc:
                if failure is None:
                    failure = exc

        if failure is not None:
            if result is not None:
                try:
                    process, thread, _process_id = result
                    self._native.abort_suspended_process(process, thread)
                except BaseException:
                    pass
            raise failure
        return result


class _CtypesWindowsLaunchNative:
    _CREATE_SUSPENDED = 0x00000004
    _CREATE_NEW_PROCESS_GROUP = 0x00000200
    _CREATE_UNICODE_ENVIRONMENT = 0x00000400
    _EXTENDED_STARTUPINFO_PRESENT = 0x00080000
    _STARTF_USESTDHANDLES = 0x00000100
    _PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
    _PROC_THREAD_ATTRIBUTE_JOB_LIST = 0x0002000D
    _DUPLICATE_SAME_ACCESS = 0x00000002
    _WAIT_OBJECT_0 = 0x00000000
    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _OPEN_EXISTING = 3
    _FILE_ATTRIBUTE_NORMAL = 0x00000080

    def __init__(self, kernel32) -> None:
        self._kernel32 = kernel32

    def get_standard_handle(self, stream: int):
        handle = self._kernel32.GetStdHandle(wintypes.DWORD(stream & 0xFFFFFFFF))
        if handle in (None, 0, ctypes.c_void_p(-1).value):
            return None
        return handle

    def open_null_handle(self, stream: int):
        desired_access = self._GENERIC_READ if stream == -10 else self._GENERIC_WRITE
        handle = self._kernel32.CreateFileW(
            "NUL",
            desired_access,
            self._FILE_SHARE_READ | self._FILE_SHARE_WRITE,
            None,
            self._OPEN_EXISTING,
            self._FILE_ATTRIBUTE_NORMAL,
            None,
        )
        if handle in (None, ctypes.c_void_p(-1).value):
            self._raise_error()
        return handle

    def duplicate_inheritable_handle(self, handle):
        current_process = self._kernel32.GetCurrentProcess()
        duplicate = wintypes.HANDLE()
        if not self._kernel32.DuplicateHandle(
            current_process,
            handle,
            current_process,
            ctypes.byref(duplicate),
            0,
            True,
            self._DUPLICATE_SAME_ACCESS,
        ):
            self._raise_error()
        return duplicate.value

    def create_attribute_list(self, handles, jobs):
        size = ctypes.c_size_t()
        self._kernel32.InitializeProcThreadAttributeList(
            None,
            2,
            0,
            ctypes.byref(size),
        )
        if size.value == 0:
            self._raise_error()
        buffer = ctypes.create_string_buffer(size.value)
        pointer = ctypes.cast(buffer, ctypes.c_void_p)
        if not self._kernel32.InitializeProcThreadAttributeList(
            pointer,
            2,
            0,
            ctypes.byref(size),
        ):
            self._raise_error()
        handle_array = (wintypes.HANDLE * len(handles))(*handles)
        if not self._kernel32.UpdateProcThreadAttribute(
            pointer,
            0,
            self._PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            ctypes.cast(handle_array, ctypes.c_void_p),
            ctypes.sizeof(handle_array),
            None,
            None,
        ):
            self._kernel32.DeleteProcThreadAttributeList(pointer)
            self._raise_error()
        job_array = (wintypes.HANDLE * len(jobs))(*jobs)
        if not self._kernel32.UpdateProcThreadAttribute(
            pointer,
            0,
            self._PROC_THREAD_ATTRIBUTE_JOB_LIST,
            ctypes.cast(job_array, ctypes.c_void_p),
            ctypes.sizeof(job_array),
            None,
            None,
        ):
            self._kernel32.DeleteProcThreadAttributeList(pointer)
            self._raise_error()
        return _Win32AttributeList(
            buffer=buffer,
            pointer=pointer,
            handle_array=handle_array,
            job_array=job_array,
        )

    def create_suspended_process(
        self,
        command: list[str],
        *,
        env: dict[str, str],
        cwd: str | None,
        standard_handles,
        attribute_list,
    ):
        command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(command))
        environment_text = "\0".join(
            f"{key}={value}"
            for key, value in sorted(
                env.items(),
                key=lambda item: item[0].casefold(),
            )
        )
        environment = ctypes.create_unicode_buffer(environment_text + "\0\0")
        startup = _STARTUPINFOEXW()
        startup.StartupInfo.cb = ctypes.sizeof(startup)
        creation_flags = (
            self._CREATE_SUSPENDED
            | self._CREATE_NEW_PROCESS_GROUP
            | self._CREATE_UNICODE_ENVIRONMENT
        )
        inherit_handles = bool(standard_handles)
        if inherit_handles:
            startup.StartupInfo.dwFlags |= self._STARTF_USESTDHANDLES
            (
                startup.StartupInfo.hStdInput,
                startup.StartupInfo.hStdOutput,
                startup.StartupInfo.hStdError,
            ) = standard_handles
            startup.lpAttributeList = attribute_list.pointer
            creation_flags |= self._EXTENDED_STARTUPINFO_PRESENT
        process_information = _PROCESS_INFORMATION()
        created = self._kernel32.CreateProcessW(
            None,
            command_line,
            None,
            None,
            inherit_handles,
            creation_flags,
            environment,
            cwd,
            ctypes.cast(ctypes.byref(startup), ctypes.POINTER(_STARTUPINFOW)),
            ctypes.byref(process_information),
        )
        if not created:
            self._raise_error()
        return (
            process_information.hProcess,
            process_information.hThread,
            int(process_information.dwProcessId),
        )

    def delete_handle_list(self, attribute_list) -> None:
        self._kernel32.DeleteProcThreadAttributeList(attribute_list.pointer)

    def close_handle(self, handle) -> None:
        if handle and not self._kernel32.CloseHandle(handle):
            self._raise_error()

    def abort_suspended_process(self, process, thread) -> None:
        failed = False
        try:
            if not self._kernel32.TerminateProcess(process, 1):
                failed = True
        except BaseException:
            failed = True
        try:
            if self._kernel32.WaitForSingleObject(process, 5000) != self._WAIT_OBJECT_0:
                failed = True
        except BaseException:
            failed = True
        for handle in (thread, process):
            try:
                if not self._kernel32.CloseHandle(handle):
                    failed = True
            except BaseException:
                failed = True
        if failed:
            self._raise_error()

    @staticmethod
    def _raise_error() -> None:
        get_last_error = getattr(ctypes, "get_last_error", None)
        raise OSError(get_last_error() if get_last_error is not None else 0)


class _Win32Api:
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
    _WAIT_OBJECT_0 = 0x00000000
    _WAIT_TIMEOUT = 0x00000102
    _INFINITE = 0xFFFFFFFF
    _CTRL_BREAK_EVENT = 1

    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32 = kernel32
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.CreateProcessW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.BOOL,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.LPCWSTR,
            ctypes.POINTER(_STARTUPINFOW),
            ctypes.POINTER(_PROCESS_INFORMATION),
        ]
        kernel32.CreateProcessW.restype = wintypes.BOOL
        kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
        kernel32.ResumeThread.restype = wintypes.DWORD
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.GenerateConsoleCtrlEvent.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel32.GenerateConsoleCtrlEvent.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel32.GetStdHandle.restype = wintypes.HANDLE
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.DuplicateHandle.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.DuplicateHandle.restype = wintypes.BOOL
        kernel32.InitializeProcThreadAttributeList.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel32.InitializeProcThreadAttributeList.restype = wintypes.BOOL
        kernel32.UpdateProcThreadAttribute.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        kernel32.UpdateProcThreadAttribute.restype = wintypes.BOOL
        kernel32.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]
        kernel32.DeleteProcThreadAttributeList.restype = None
        self._process_launcher = _WindowsProcessLauncher(
            _CtypesWindowsLaunchNative(kernel32)
        )

    def create_job(self):
        job = self._kernel32.CreateJobObjectW(None, None)
        if not job:
            self._raise_error()
        return job

    def set_kill_on_close(self, job) -> None:
        information = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        information.BasicLimitInformation.LimitFlags = (
            self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not self._kernel32.SetInformationJobObject(
            job,
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            self._raise_error()

    def create_suspended_process(
        self,
        command: list[str],
        *,
        env: dict[str, str],
        cwd: str | None,
        job,
    ):
        return self._process_launcher.create(
            command,
            env=env,
            cwd=cwd,
            job=job,
        )

    def resume_thread(self, thread) -> None:
        previous_count = self._kernel32.ResumeThread(thread)
        if previous_count != 1:
            self._raise_error()

    def terminate_job(self, job) -> None:
        if not self._kernel32.TerminateJobObject(job, 1):
            self._raise_error()

    def wait_process(self, process, timeout: float | None) -> bool:
        milliseconds = (
            self._INFINITE
            if timeout is None
            else min(self._INFINITE - 1, max(0, math.ceil(timeout * 1000)))
        )
        result = self._kernel32.WaitForSingleObject(process, milliseconds)
        if result == self._WAIT_OBJECT_0:
            return True
        if result == self._WAIT_TIMEOUT:
            return False
        self._raise_error()

    def process_exit_code(self, process) -> int:
        exit_code = wintypes.DWORD()
        if not self._kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code)):
            self._raise_error()
        return int(exit_code.value)

    def wait_for_job_empty(self, job, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            information = _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
            if not self._kernel32.QueryInformationJobObject(
                job,
                self._JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
                ctypes.byref(information),
                ctypes.sizeof(information),
                None,
            ):
                self._raise_error()
            if information.ActiveProcesses == 0:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.02, remaining))

    def send_ctrl_break(self, process_id: int) -> None:
        if not self._kernel32.GenerateConsoleCtrlEvent(
            self._CTRL_BREAK_EVENT,
            process_id,
        ):
            self._raise_error()

    def close_handle(self, handle) -> None:
        if handle and not self._kernel32.CloseHandle(handle):
            self._raise_error()

    @staticmethod
    def _raise_error() -> None:
        raise OSError(ctypes.get_last_error())


class _WindowsChild:
    def __init__(
        self,
        *,
        api: _Win32ApiProtocol,
        job,
        process,
        process_id: int,
    ) -> None:
        self._api = api
        self._job = job
        self._process = process
        self._process_id = process_id

    @classmethod
    def start(
        cls,
        command: list[str],
        *,
        env: dict[str, str],
        cwd: str | None,
        api: _Win32ApiProtocol | None = None,
    ) -> Self:
        native = api or _Win32Api()
        job = None
        process = None
        thread = None
        try:
            job = native.create_job()
            native.set_kill_on_close(job)
            process, thread, process_id = native.create_suspended_process(
                command,
                env=env,
                cwd=cwd,
                job=job,
            )
            native.resume_thread(thread)
            native.close_handle(thread)
            thread = None
            return cls(
                api=native,
                job=job,
                process=process,
                process_id=process_id,
            )
        except BaseException as exc:
            cls._rollback_start(
                native,
                job=job,
                process=process,
                thread=thread,
            )
            raise ProcessSupervisionError() from exc

    @staticmethod
    def _rollback_start(
        api: _Win32ApiProtocol,
        *,
        job,
        process,
        thread,
    ) -> None:
        if process is not None:
            if job is not None:
                try:
                    api.terminate_job(job)
                except BaseException:
                    pass
                try:
                    api.wait_for_job_empty(job, 5.0)
                except BaseException:
                    pass
        for handle in (thread, process, job):
            if handle is None:
                continue
            try:
                api.close_handle(handle)
            except BaseException:
                pass

    def wait(self) -> int:
        try:
            while not self._api.wait_process(
                self._process,
                _WINDOWS_WAIT_POLL_SECONDS,
            ):
                pass
            return self._api.process_exit_code(self._process)
        except (KeyboardInterrupt, _ForwardedSignal):
            raise
        except ProcessSupervisionError:
            raise
        except BaseException as exc:
            raise ProcessSupervisionError() from exc

    def forward_signal(self, signum: int) -> None:
        try:
            if signum == signal.SIGINT:
                try:
                    self._api.send_ctrl_break(self._process_id)
                except BaseException:
                    self._api.terminate_job(self._job)
            else:
                self._api.terminate_job(self._job)
        except BaseException as exc:
            raise ProcessSupervisionError() from exc

    def forward_and_reap(self, signum: int, *, timeout: float) -> None:
        try:
            if signum == signal.SIGINT:
                try:
                    self._api.send_ctrl_break(self._process_id)
                except BaseException:
                    self._api.terminate_job(self._job)
                    deadline = time.monotonic() + timeout
                    job_empty = self._api.wait_for_job_empty(self._job, timeout)
                else:
                    deadline = time.monotonic() + timeout
                    job_empty = self._api.wait_for_job_empty(self._job, timeout)
                    if not job_empty:
                        self._api.terminate_job(self._job)
                        deadline = time.monotonic() + timeout
                        job_empty = self._api.wait_for_job_empty(
                            self._job,
                            timeout,
                        )
            else:
                self._api.terminate_job(self._job)
                deadline = time.monotonic() + timeout
                job_empty = self._api.wait_for_job_empty(self._job, timeout)
            if not job_empty:
                raise ProcessSupervisionError()
            if not self._wait_process_until(deadline):
                raise ProcessSupervisionError()
        except ProcessSupervisionError:
            raise
        except BaseException as exc:
            raise ProcessSupervisionError() from exc

    def terminate_and_reap(self, *, timeout: float) -> None:
        try:
            self._api.terminate_job(self._job)
            deadline = time.monotonic() + timeout
            if not self._api.wait_for_job_empty(self._job, timeout):
                raise ProcessSupervisionError()
            if not self._wait_process_until(deadline):
                raise ProcessSupervisionError()
        except ProcessSupervisionError:
            raise
        except BaseException as exc:
            raise ProcessSupervisionError() from exc

    def close_remaining_tree(self, *, timeout: float) -> None:
        try:
            if self._api.wait_for_job_empty(self._job, 0.0):
                return
            self._api.terminate_job(self._job)
            if not self._api.wait_for_job_empty(self._job, timeout):
                raise ProcessSupervisionError()
        except ProcessSupervisionError:
            raise
        except BaseException as exc:
            raise ProcessSupervisionError() from exc

    def ensure_closed(self, *, timeout: float) -> None:
        self.close_remaining_tree(timeout=timeout)

    def close(self) -> None:
        failed = False
        process, self._process = self._process, None
        job, self._job = self._job, None
        for handle in (process, job):
            if handle is None:
                continue
            try:
                self._api.close_handle(handle)
            except BaseException:
                failed = True
        if failed:
            raise ProcessSupervisionError()

    def _wait_process_until(self, deadline: float) -> bool:
        while True:
            remaining = deadline - time.monotonic()
            wait_seconds = max(
                0.0,
                min(_WINDOWS_WAIT_POLL_SECONDS, remaining),
            )
            if self._api.wait_process(self._process, wait_seconds):
                return True
            if remaining <= 0:
                return False


class ForegroundChildSupervisor:
    def __init__(self, child: _PosixChild | _WindowsChild) -> None:
        self._child = child

    @classmethod
    def start(
        cls,
        command: list[str],
        *,
        env: dict[str, str],
        cwd: str | None = None,
    ) -> Self:
        try:
            child = (
                _WindowsChild.start(command, env=env, cwd=cwd)
                if _IS_WINDOWS
                else _PosixChild.start(command, env=env, cwd=cwd)
            )
        except ProcessSupervisionError:
            raise
        except BaseException as exc:
            raise ProcessSupervisionError() from exc
        return cls(child)

    def wait(self) -> int:
        return self._child.wait()

    def forward_signal(self, signum: int) -> None:
        self._child.forward_signal(signum)

    def forward_and_reap(self, signum: int, *, timeout: float) -> None:
        self._child.forward_and_reap(signum, timeout=timeout)

    def terminate_and_reap(self, *, timeout: float) -> None:
        self._child.terminate_and_reap(timeout=timeout)

    def close_remaining_tree(self, *, timeout: float) -> None:
        self._child.close_remaining_tree(timeout=timeout)

    def ensure_closed(self, *, timeout: float) -> None:
        self._child.ensure_closed(timeout=timeout)

    def close(self) -> None:
        self._child.close()
