from __future__ import annotations

import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest


class _FakePopen:
    def __init__(
        self,
        *,
        pid: int = 4312,
        wait_results: list[int | BaseException] | None = None,
    ) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.wait_results = list(wait_results or [0])
        self.wait_timeouts: list[float | None] = []

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if not self.wait_results:
            assert self.returncode is not None
            return self.returncode
        result = self.wait_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        self.returncode = result
        return result

    def poll(self) -> int | None:
        return self.returncode


class _SignalHarness:
    def __init__(
        self,
        *,
        deliver_on_restore: int | None = None,
        deliver_on_install: int | None = None,
        old_mask: frozenset[int] | None = None,
    ) -> None:
        self.previous = {
            signal.SIGINT: object(),
            signal.SIGTERM: object(),
        }
        self.handlers = dict(self.previous)
        self.old_mask = (
            old_mask
            if old_mask is not None
            else (
                frozenset({signal.SIGUSR1})
                if hasattr(signal, "SIGUSR1")
                else frozenset()
            )
        )
        self.deliver_on_restore = deliver_on_restore
        self.deliver_on_install = deliver_on_install
        self.events: list[tuple[str, object]] = []

    def getsignal(self, signum: int):
        return self.handlers[signum]

    def install(self, signum: int, handler):
        previous = self.handlers[signum]
        self.handlers[signum] = handler
        self.events.append(("handler", signum))
        if self.deliver_on_install == signum:
            self.deliver_on_install = None
            handler(signum, None)
        return previous

    def pthread_sigmask(self, operation: int, mask):
        frozen_mask = frozenset(mask)
        self.events.append(("mask", (operation, frozen_mask)))
        if operation == signal.SIG_BLOCK:
            return self.old_mask
        assert operation == signal.SIG_SETMASK
        assert frozen_mask == self.old_mask
        if self.deliver_on_restore is not None:
            signum = self.deliver_on_restore
            self.deliver_on_restore = None
            self.handlers[signum](signum, None)
        return frozenset({signal.SIGINT, signal.SIGTERM})


class _AttachedChild:
    def __init__(self) -> None:
        self.forwarded: list[int] = []

    def forward_signal(self, signum: int) -> None:
        self.forwarded.append(signum)


def _install_signal_harness(
    monkeypatch: pytest.MonkeyPatch,
    harness: _SignalHarness,
    *,
    is_windows: bool = False,
):
    from agentseek_api import process_supervisor as supervisor_module

    monkeypatch.setattr(supervisor_module, "_IS_WINDOWS", is_windows)
    monkeypatch.setattr(supervisor_module.signal, "getsignal", harness.getsignal)
    monkeypatch.setattr(supervisor_module.signal, "signal", harness.install)
    monkeypatch.setattr(
        supervisor_module.signal,
        "pthread_sigmask",
        harness.pthread_sigmask,
    )
    return supervisor_module


def test_forwarding_signal_guard_restores_exact_handlers_and_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _SignalHarness()
    supervisor_module = _install_signal_harness(monkeypatch, harness)

    with pytest.raises(RuntimeError, match="body-canary"):
        with supervisor_module.ForwardingSignalGuard():
            raise RuntimeError("body-canary")

    assert harness.handlers == harness.previous
    assert harness.events[0] == (
        "mask",
        (
            signal.SIG_BLOCK,
            frozenset({signal.SIGINT, signal.SIGTERM}),
        ),
    )
    assert harness.events[-1] == (
        "mask",
        (signal.SIG_SETMASK, harness.old_mask),
    )


def test_pending_signal_is_delivered_only_after_child_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _SignalHarness(deliver_on_restore=signal.SIGTERM)
    supervisor_module = _install_signal_harness(monkeypatch, harness)
    child = _AttachedChild()

    with pytest.raises(supervisor_module._ForwardedSignal) as captured:
        with supervisor_module.ForwardingSignalGuard() as guard:
            assert child.forwarded == []
            guard.attach(child)

    assert captured.value.signum == signal.SIGTERM
    assert harness.handlers == harness.previous


def test_second_signal_during_cleanup_is_non_throwing_and_reforwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _SignalHarness()
    supervisor_module = _install_signal_harness(monkeypatch, harness)
    child = _AttachedChild()

    with supervisor_module.ForwardingSignalGuard() as guard:
        guard.attach(child)
        guard.begin_cleanup()
        harness.handlers[signal.SIGTERM](signal.SIGTERM, None)
        harness.handlers[signal.SIGTERM](signal.SIGTERM, None)

    assert child.forwarded == [signal.SIGTERM, signal.SIGTERM]
    assert harness.handlers == harness.previous


def test_guard_rejects_unverified_handler_installation_and_restores_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _SignalHarness()
    supervisor_module = _install_signal_harness(monkeypatch, harness)

    def ignore_sigterm_install(signum: int, handler):
        previous = harness.handlers[signum]
        if signum == signal.SIGINT or handler in harness.previous.values():
            harness.handlers[signum] = handler
        return previous

    monkeypatch.setattr(
        supervisor_module.signal,
        "signal",
        ignore_sigterm_install,
    )

    with pytest.raises(supervisor_module.ProcessSupervisionError):
        with supervisor_module.ForwardingSignalGuard():
            pass

    assert harness.handlers == harness.previous


def test_guard_fails_closed_when_callers_mask_blocks_forwarded_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _SignalHarness(old_mask=frozenset({signal.SIGTERM}))
    supervisor_module = _install_signal_harness(monkeypatch, harness)

    with pytest.raises(supervisor_module.ProcessSupervisionError):
        with supervisor_module.ForwardingSignalGuard():
            pass

    assert harness.handlers == harness.previous
    assert (
        "mask",
        (signal.SIG_SETMASK, harness.old_mask),
    ) in harness.events


def test_signal_arriving_during_popen_is_pending_until_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _SignalHarness()
    supervisor_module = _install_signal_harness(monkeypatch, harness)
    process = _FakePopen(wait_results=[0])

    def fake_popen(command, **kwargs):
        harness.handlers[signal.SIGTERM](signal.SIGTERM, None)
        return process

    monkeypatch.setattr(supervisor_module.subprocess, "Popen", fake_popen)

    with pytest.raises(supervisor_module._ForwardedSignal) as captured:
        with supervisor_module.ForwardingSignalGuard() as guard:
            child = supervisor_module.ForegroundChildSupervisor.start(
                ["child"],
                env={},
                cwd=None,
            )
            guard.attach(child)

    assert captured.value.signum == signal.SIGTERM


def test_windows_signal_during_first_handler_install_is_not_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _SignalHarness(deliver_on_install=signal.SIGINT)
    supervisor_module = _install_signal_harness(
        monkeypatch,
        harness,
        is_windows=True,
    )

    with pytest.raises(supervisor_module._ForwardedSignal) as captured:
        with supervisor_module.ForwardingSignalGuard() as guard:
            guard.attach(_AttachedChild())

    assert captured.value.signum == signal.SIGINT


def test_signal_during_pending_consumption_is_delivered_from_attach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _SignalHarness()
    supervisor_module = _install_signal_harness(monkeypatch, harness)

    class _AttachRaceGuard(supervisor_module.ForwardingSignalGuard):
        def __init__(self) -> None:
            self.arm_pending_read = False
            super().__init__()

        def __getattribute__(self, name: str):
            value = object.__getattribute__(self, name)
            if name == "_pending_signal" and object.__getattribute__(
                self, "arm_pending_read"
            ):
                object.__setattr__(self, "arm_pending_read", False)
                object.__getattribute__(self, "_installed_handler")(
                    signal.SIGTERM,
                    None,
                )
            return value

    guard = _AttachRaceGuard()
    with pytest.raises(supervisor_module._ForwardedSignal) as captured:
        with guard:
            harness.handlers[signal.SIGINT](signal.SIGINT, None)
            guard.arm_pending_read = True
            guard.attach(_AttachedChild())

    assert captured.value.signum == signal.SIGINT


def test_posix_start_uses_new_session_without_shell_and_preserves_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import process_supervisor as supervisor_module

    process = _FakePopen(wait_results=[23])
    observed: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        observed.update(command=command, kwargs=kwargs)
        return process

    monkeypatch.setattr(supervisor_module, "_IS_WINDOWS", False)
    monkeypatch.setattr(supervisor_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        supervisor_module,
        "_waitid_no_reap",
        lambda _pid, *, nohang: 23,
    )
    monkeypatch.setattr(
        supervisor_module,
        "_process_group_has_other_members",
        lambda _pgid, _leader_pid: False,
    )
    command = ["python", "command-canary"]
    environment = {"SECRET": "environment-canary"}

    child = supervisor_module.ForegroundChildSupervisor.start(
        command,
        env=environment,
        cwd="/runtime",
    )

    assert child.wait() == 23
    child.close_remaining_tree(timeout=5.0)
    child.ensure_closed(timeout=5.0)
    child.close()
    assert process.wait_timeouts == [0.0]
    assert observed == {
        "command": command,
        "kwargs": {
            "env": environment,
            "cwd": "/runtime",
            "start_new_session": True,
        },
    }


def test_posix_start_failure_is_value_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import process_supervisor as supervisor_module

    monkeypatch.setattr(supervisor_module, "_IS_WINDOWS", False)

    def fail_popen(command, **kwargs):
        raise OSError("setup-canary command-canary environment-canary")

    monkeypatch.setattr(supervisor_module.subprocess, "Popen", fail_popen)

    with pytest.raises(supervisor_module.ProcessSupervisionError) as captured:
        supervisor_module.ForegroundChildSupervisor.start(
            ["command-canary"],
            env={"SECRET": "environment-canary"},
            cwd=None,
        )

    assert "setup-canary" not in str(captured.value)
    assert "command-canary" not in str(captured.value)
    assert "environment-canary" not in str(captured.value)


def test_posix_start_fails_before_popen_without_nonreaping_wait_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import process_supervisor as supervisor_module

    popen_called = False

    def fake_popen(command, **kwargs):
        nonlocal popen_called
        popen_called = True
        return _FakePopen()

    monkeypatch.setattr(supervisor_module.sys, "platform", "linux")
    monkeypatch.delattr(supervisor_module.os, "WNOWAIT")
    monkeypatch.setattr(supervisor_module.subprocess, "Popen", fake_popen)

    with pytest.raises(supervisor_module.ProcessSupervisionError):
        supervisor_module._PosixChild.start(["child"], env={}, cwd=None)

    assert popen_called is False


def test_darwin_without_os_waitid_uses_native_nonreaping_observer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import process_supervisor as supervisor_module

    observed: list[tuple[int, bool]] = []
    monkeypatch.setattr(supervisor_module.sys, "platform", "darwin")
    monkeypatch.delattr(supervisor_module.os, "waitid")
    monkeypatch.setattr(
        supervisor_module,
        "_darwin_waitid_no_reap",
        lambda pid, *, nohang: observed.append((pid, nohang)) or 27,
        raising=False,
    )

    assert supervisor_module._waitid_no_reap(4312, nohang=True) == 27
    assert observed == [(4312, True)]


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin libc waitid only")
def test_darwin_native_waitid_observes_exit_without_reaping() -> None:
    from agentseek_api import process_supervisor as supervisor_module

    process = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(27)"],
    )
    try:
        assert (
            supervisor_module._darwin_waitid_no_reap(
                process.pid,
                nohang=False,
            )
            == 27
        )
        assert process.returncode is None
        assert process.wait(timeout=1.0) == 27
    finally:
        if process.returncode is None:
            process.kill()
            process.wait(timeout=1.0)


def test_posix_waitid_preserves_forwarded_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import process_supervisor as supervisor_module

    forwarded = supervisor_module._ForwardedSignal(signal.SIGTERM)
    monkeypatch.setattr(
        supervisor_module.os,
        "waitid",
        lambda id_type, pid, options: (_ for _ in ()).throw(forwarded),
    )

    with pytest.raises(supervisor_module._ForwardedSignal) as captured:
        supervisor_module._waitid_no_reap(4312, nohang=False)

    assert captured.value is forwarded


def test_posix_persistent_observer_failure_still_attempts_final_direct_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import process_supervisor as supervisor_module

    process = _FakePopen(wait_results=[-signal.SIGKILL])
    signals: list[int] = []
    deadlines: list[float] = []
    monkeypatch.setattr(
        supervisor_module.subprocess,
        "Popen",
        lambda command, **kwargs: process,
    )
    monkeypatch.setattr(
        supervisor_module,
        "_waitid_no_reap",
        lambda pid, *, nohang: (_ for _ in ()).throw(
            supervisor_module.ProcessSupervisionError()
        ),
    )
    monkeypatch.setattr(supervisor_module.os, "getpgid", lambda _pid: process.pid)
    monkeypatch.setattr(
        supervisor_module.os,
        "killpg",
        lambda _pgid, signum: signals.append(signum),
    )
    child = supervisor_module.ForegroundChildSupervisor.start(
        ["child"],
        env={},
        cwd=None,
    )

    def wait_for_tree(*, deadline: float) -> tuple[bool, bool]:
        deadlines.append(deadline)
        return False, True

    monkeypatch.setattr(child._child, "_wait_for_owned_tree_exit", wait_for_tree)

    with pytest.raises(supervisor_module.ProcessSupervisionError):
        child.forward_and_reap(signal.SIGTERM, timeout=5.0)

    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert len(deadlines) == 2
    assert process.wait_timeouts == [0.0]
    with pytest.raises(supervisor_module.ProcessSupervisionError):
        child.ensure_closed(timeout=5.0)
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_posix_group_forwarding_escalates_and_reaps_direct_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import process_supervisor as supervisor_module

    process = _FakePopen(wait_results=[-signal.SIGKILL])
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(supervisor_module, "_IS_WINDOWS", False)
    monkeypatch.setattr(
        supervisor_module.subprocess,
        "Popen",
        lambda command, **kwargs: process,
    )
    monkeypatch.setattr(
        supervisor_module.os,
        "killpg",
        lambda pgid, signum: sent.append((pgid, signum)),
    )
    monkeypatch.setattr(
        supervisor_module,
        "_waitid_no_reap",
        lambda _pid, *, nohang: None,
    )
    monkeypatch.setattr(supervisor_module.os, "getpgid", lambda _pid: process.pid)

    child = supervisor_module.ForegroundChildSupervisor.start(
        ["child"],
        env={},
        cwd=None,
    )
    outcomes = iter([(False, False), (True, False)])

    def wait_for_tree(*, deadline: float) -> tuple[bool, bool]:
        outcome = next(outcomes)
        if outcome[0]:
            child._child._observed_exit_code = -signal.SIGKILL
        return outcome

    monkeypatch.setattr(child._child, "_wait_for_owned_tree_exit", wait_for_tree)
    child.forward_and_reap(signal.SIGTERM, timeout=5.0)

    assert sent == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
    assert process.wait_timeouts == [0.0]


def test_posix_hard_kill_wait_is_bounded_when_direct_child_does_not_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import process_supervisor as supervisor_module

    process = _FakePopen(wait_results=[subprocess.TimeoutExpired("child", 0.0)])
    monkeypatch.setattr(supervisor_module, "_IS_WINDOWS", False)
    monkeypatch.setattr(
        supervisor_module.subprocess,
        "Popen",
        lambda command, **kwargs: process,
    )
    monkeypatch.setattr(supervisor_module.os, "killpg", lambda pgid, signum: None)
    monkeypatch.setattr(
        supervisor_module,
        "_waitid_no_reap",
        lambda _pid, *, nohang: None,
    )
    monkeypatch.setattr(supervisor_module.os, "getpgid", lambda _pid: process.pid)
    child = supervisor_module.ForegroundChildSupervisor.start(
        ["child"],
        env={},
        cwd=None,
    )
    deadlines: list[float] = []

    def wait_for_tree(*, deadline: float) -> tuple[bool, bool]:
        deadlines.append(deadline)
        return False, False

    monkeypatch.setattr(child._child, "_wait_for_owned_tree_exit", wait_for_tree)

    with pytest.raises(supervisor_module.ProcessSupervisionError):
        child.forward_and_reap(signal.SIGTERM, timeout=5.0)

    assert len(deadlines) == 2
    assert all(deadline < float("inf") for deadline in deadlines)
    assert process.wait_timeouts == [0.0]


def test_posix_hard_cleanup_uses_a_separate_finite_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import process_supervisor as supervisor_module

    process = _FakePopen(wait_results=[-signal.SIGKILL])
    monotonic_values = iter([0.0, 10.0])
    deadlines: list[float] = []
    monkeypatch.setattr(
        supervisor_module.subprocess,
        "Popen",
        lambda command, **kwargs: process,
    )
    monkeypatch.setattr(supervisor_module.os, "killpg", lambda pgid, signum: None)
    monkeypatch.setattr(supervisor_module.os, "getpgid", lambda _pid: process.pid)
    monkeypatch.setattr(
        supervisor_module,
        "_waitid_no_reap",
        lambda _pid, *, nohang: None,
    )
    monkeypatch.setattr(
        supervisor_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    child = supervisor_module.ForegroundChildSupervisor.start(
        ["child"],
        env={},
        cwd=None,
    )

    def wait_for_tree(*, deadline: float) -> tuple[bool, bool]:
        deadlines.append(deadline)
        if len(deadlines) == 2:
            child._child._observed_exit_code = -signal.SIGKILL
            return True, False
        return False, False

    monkeypatch.setattr(child._child, "_wait_for_owned_tree_exit", wait_for_tree)

    child.forward_and_reap(signal.SIGTERM, timeout=5.0)

    assert deadlines == [5.0, 15.0]
    assert process.wait_timeouts == [0.0]


def test_posix_normal_return_terminates_remaining_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import process_supervisor as supervisor_module

    process = _FakePopen(wait_results=[29])
    group_states = iter([True, False])
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(supervisor_module, "_IS_WINDOWS", False)
    monkeypatch.setattr(
        supervisor_module.subprocess,
        "Popen",
        lambda command, **kwargs: process,
    )
    monkeypatch.setattr(
        supervisor_module,
        "_waitid_no_reap",
        lambda _pid, *, nohang: 29,
    )
    monkeypatch.setattr(
        supervisor_module,
        "_process_group_has_other_members",
        lambda _pgid, _leader_pid: next(group_states),
    )
    monkeypatch.setattr(supervisor_module.os, "getpgid", lambda _pid: process.pid)
    monkeypatch.setattr(
        supervisor_module.os,
        "killpg",
        lambda pgid, signum: sent.append((pgid, signum)),
    )

    child = supervisor_module.ForegroundChildSupervisor.start(
        ["child"],
        env={},
        cwd=None,
    )

    assert child.wait() == 29
    child.close_remaining_tree(timeout=5.0)
    assert sent == [(process.pid, signal.SIGTERM)]
    assert process.wait_timeouts == [0.0]


def test_posix_normal_return_retains_leader_until_group_cleanup_then_reaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import process_supervisor as supervisor_module

    process = _FakePopen(wait_results=[41])
    events: list[tuple[str, object]] = []
    other_member_states = iter([True, False])
    monkeypatch.setattr(
        supervisor_module.subprocess,
        "Popen",
        lambda command, **kwargs: process,
    )

    def waitid(id_type, pid, options):
        events.append(("observe", (id_type, pid, options)))
        return SimpleNamespace(
            si_pid=pid,
            si_code=supervisor_module.os.CLD_EXITED,
            si_status=41,
        )

    monkeypatch.setattr(supervisor_module.os, "waitid", waitid)
    monkeypatch.setattr(
        supervisor_module.os,
        "getpgid",
        lambda _pid: (_ for _ in ()).throw(ProcessLookupError()),
    )

    def has_other_members(pgid: int, leader_pid: int) -> bool:
        events.append(("members", (pgid, leader_pid)))
        return next(other_member_states)

    monkeypatch.setattr(
        supervisor_module,
        "_process_group_has_other_members",
        has_other_members,
        raising=False,
    )
    monkeypatch.setattr(
        supervisor_module.os,
        "killpg",
        lambda pgid, signum: events.append(("signal", (pgid, signum))),
    )
    child = supervisor_module.ForegroundChildSupervisor.start(
        ["child"],
        env={},
        cwd=None,
    )

    assert child.wait() == 41
    assert process.wait_timeouts == []
    child.close_remaining_tree(timeout=5.0)

    assert process.wait_timeouts == [0.0]
    assert [name for name, _value in events] == [
        "observe",
        "members",
        "signal",
        "members",
    ]
    assert events[2] == ("signal", (process.pid, signal.SIGTERM))


class _FakeWin32Api:
    def __init__(
        self,
        *,
        fail_at: str | None = None,
        exit_code: int = 0,
        job_empty_results: list[bool] | None = None,
        wait_process_results: list[bool | BaseException] | None = None,
    ) -> None:
        self.fail_at = fail_at
        self.failed = False
        self.exit_code = exit_code
        self.job_empty_results = list(job_empty_results or [True])
        self.wait_process_results = list(wait_process_results or [True])
        self.events: list[tuple[str, object]] = []

    def _record(self, name: str, value: object = None) -> None:
        self.events.append((name, value))
        if self.fail_at == name and not self.failed:
            self.failed = True
            raise OSError(f"{name}-setup-canary")

    def create_job(self):
        self._record("create-job")
        return "job-handle"

    def set_kill_on_close(self, job) -> None:
        self._record("set-kill-on-close", job)

    def create_suspended_process(self, command, *, env, cwd):
        self._record(
            "create-suspended",
            (list(command), dict(env), cwd),
        )
        return "process-handle", "thread-handle", 8128

    def assign_process_to_job(self, job, process) -> None:
        self._record("assign-job", (job, process))

    def resume_thread(self, thread) -> None:
        self._record("resume-thread", thread)

    def terminate_process(self, process) -> None:
        self._record("terminate-process", process)

    def terminate_job(self, job) -> None:
        self._record("terminate-job", job)

    def wait_process(self, process, timeout: float | None) -> bool:
        self._record("wait-process", (process, timeout))
        result = self.wait_process_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def process_exit_code(self, process) -> int:
        self._record("exit-code", process)
        return self.exit_code

    def wait_for_job_empty(self, job, timeout: float) -> bool:
        self._record("wait-job-empty", (job, timeout))
        return self.job_empty_results.pop(0)

    def send_ctrl_break(self, process_id: int) -> None:
        self._record("ctrl-break", process_id)

    def close_handle(self, handle) -> None:
        self._record(f"close-{handle}", handle)


class _FakeWindowsLaunchNative:
    def __init__(
        self,
        *,
        missing_streams: frozenset[int] = frozenset(),
        fail_duplicate_at: int | None = None,
        fail_delete: bool = False,
        fail_close: frozenset[str] = frozenset(),
        fail_abort: bool = False,
    ) -> None:
        self.unrelated_inheritable_handle = "sentinel-handle"
        self.missing_streams = missing_streams
        self.fail_duplicate_at = fail_duplicate_at
        self.fail_delete = fail_delete
        self.fail_close = fail_close
        self.fail_abort = fail_abort
        self.duplicate_calls = 0
        self.events: list[tuple[str, object]] = []

    def get_standard_handle(self, stream: int):
        if stream in self.missing_streams:
            self.events.append(("get-standard", (stream, None)))
            return None
        handle = {
            -10: "stdin-handle",
            -11: "stdout-handle",
            -12: "stderr-handle",
        }[stream]
        self.events.append(("get-standard", (stream, handle)))
        return handle

    def open_null_handle(self, stream: int):
        handle = f"null-handle-{stream}"
        self.events.append(("open-null", (stream, handle)))
        return handle

    def duplicate_inheritable_handle(self, handle):
        duplicate_index = self.duplicate_calls
        self.duplicate_calls += 1
        if duplicate_index == self.fail_duplicate_at:
            self.events.append(("duplicate-failed", handle))
            raise OSError("duplicate-canary")
        duplicate = f"duplicate-{handle}"
        self.events.append(("duplicate", (handle, duplicate)))
        return duplicate

    def create_handle_list(self, handles):
        self.events.append(("create-handle-list", tuple(handles)))
        return "attribute-list"

    def create_suspended_process(
        self,
        command,
        *,
        env,
        cwd,
        standard_handles,
        attribute_list,
    ):
        self.events.append(
            (
                "create-process",
                {
                    "command": list(command),
                    "env": dict(env),
                    "cwd": cwd,
                    "standard_handles": tuple(standard_handles),
                    "attribute_list": attribute_list,
                },
            )
        )
        return "process-handle", "thread-handle", 9127

    def delete_handle_list(self, attribute_list) -> None:
        self.events.append(("delete-handle-list", attribute_list))
        if self.fail_delete:
            raise OSError("delete-canary")

    def close_handle(self, handle) -> None:
        self.events.append(("close-duplicate", handle))
        if handle in self.fail_close:
            raise OSError("close-canary")

    def abort_suspended_process(self, process, thread) -> None:
        self.events.append(("abort-process", (process, thread)))
        if self.fail_abort:
            raise OSError("abort-canary")


def test_windows_launcher_inherits_only_inheritable_stdio_duplicates() -> None:
    from agentseek_api import process_supervisor as supervisor_module

    native = _FakeWindowsLaunchNative()
    launcher = supervisor_module._WindowsProcessLauncher(native)

    result = launcher.create(
        ["python", "child.py"],
        env={"TOKEN": "value"},
        cwd="C:\\runtime",
    )

    assert result == ("process-handle", "thread-handle", 9127)
    create_event = next(
        value for name, value in native.events if name == "create-process"
    )
    expected_duplicates = (
        "duplicate-stdin-handle",
        "duplicate-stdout-handle",
        "duplicate-stderr-handle",
    )
    assert create_event["standard_handles"] == expected_duplicates
    assert native.unrelated_inheritable_handle not in create_event["standard_handles"]
    assert ("create-handle-list", expected_duplicates) in native.events
    assert native.events[-4:] == [
        ("delete-handle-list", "attribute-list"),
        ("close-duplicate", "duplicate-stdin-handle"),
        ("close-duplicate", "duplicate-stdout-handle"),
        ("close-duplicate", "duplicate-stderr-handle"),
    ]


@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_windows_launcher_closes_partial_standard_handle_duplicates(
    failure_index: int,
) -> None:
    from agentseek_api import process_supervisor as supervisor_module

    native = _FakeWindowsLaunchNative(fail_duplicate_at=failure_index)
    launcher = supervisor_module._WindowsProcessLauncher(native)

    with pytest.raises(OSError, match="duplicate-canary"):
        launcher.create(["child"], env={}, cwd=None)

    closed_duplicates = [
        value
        for name, value in native.events
        if name == "close-duplicate" and str(value).startswith("duplicate-")
    ]
    assert (
        closed_duplicates
        == [
            "duplicate-stdin-handle",
            "duplicate-stdout-handle",
        ][:failure_index]
    )


def test_windows_launcher_substitutes_null_for_missing_stdin() -> None:
    from agentseek_api import process_supervisor as supervisor_module

    native = _FakeWindowsLaunchNative(missing_streams=frozenset({-10}))
    launcher = supervisor_module._WindowsProcessLauncher(native)

    launcher.create(["child"], env={}, cwd=None)

    create_event = next(
        value for name, value in native.events if name == "create-process"
    )
    assert create_event["standard_handles"] == (
        "duplicate-null-handle--10",
        "duplicate-stdout-handle",
        "duplicate-stderr-handle",
    )
    assert (
        "create-handle-list",
        create_event["standard_handles"],
    ) in native.events
    assert ("close-duplicate", "null-handle--10") in native.events


def test_windows_launcher_attempts_all_cleanup_before_aborting_created_process() -> (
    None
):
    from agentseek_api import process_supervisor as supervisor_module

    native = _FakeWindowsLaunchNative(
        fail_delete=True,
        fail_close=frozenset({"duplicate-stdout-handle"}),
        fail_abort=True,
    )
    launcher = supervisor_module._WindowsProcessLauncher(native)

    with pytest.raises(OSError, match="delete-canary"):
        launcher.create(["child"], env={}, cwd=None)

    assert ("abort-process", ("process-handle", "thread-handle")) in native.events
    assert [value for name, value in native.events if name == "close-duplicate"] == [
        "duplicate-stdin-handle",
        "duplicate-stdout-handle",
        "duplicate-stderr-handle",
    ]


class _FakeAbortKernel32:
    def __init__(self, failure_point: str) -> None:
        self.failure_point = failure_point
        self.events: list[tuple[str, object]] = []

    def TerminateProcess(self, process, exit_code: int) -> bool:
        self.events.append(("terminate", (process, exit_code)))
        return self.failure_point != "terminate"

    def WaitForSingleObject(self, process, timeout: int) -> int:
        self.events.append(("wait", (process, timeout)))
        if self.failure_point == "wait-timeout":
            return 258
        if self.failure_point == "wait-failed":
            return 0xFFFFFFFF
        return 0

    def CloseHandle(self, handle) -> bool:
        self.events.append(("close", handle))
        return self.failure_point != f"close-{handle}"


@pytest.mark.parametrize(
    "failure_point",
    [
        "terminate",
        "wait-timeout",
        "wait-failed",
        "close-thread-handle",
        "close-process-handle",
    ],
)
def test_windows_native_abort_reports_failures_after_attempting_all_cleanup(
    failure_point: str,
) -> None:
    from agentseek_api import process_supervisor as supervisor_module

    kernel32 = _FakeAbortKernel32(failure_point)
    native = supervisor_module._CtypesWindowsLaunchNative(kernel32)

    with pytest.raises(OSError):
        native.abort_suspended_process("process-handle", "thread-handle")

    assert kernel32.events == [
        ("terminate", ("process-handle", 1)),
        ("wait", ("process-handle", 5000)),
        ("close", "thread-handle"),
        ("close", "process-handle"),
    ]


def test_windows_child_is_assigned_to_kill_on_close_job_before_resume() -> None:
    from agentseek_api import process_supervisor as supervisor_module

    api = _FakeWin32Api()
    child = supervisor_module._WindowsChild.start(
        ["python", "command-canary"],
        env={"SECRET": "environment-canary"},
        cwd="C:\\runtime",
        api=api,
    )
    child.close()

    names = [name for name, _value in api.events]
    assert names[:6] == [
        "create-job",
        "set-kill-on-close",
        "create-suspended",
        "assign-job",
        "resume-thread",
        "close-thread-handle",
    ]
    assert names.index("assign-job") < names.index("resume-thread")
    assert names[-2:] == ["close-process-handle", "close-job-handle"]


@pytest.mark.parametrize(
    "failure_point",
    [
        "set-kill-on-close",
        "create-suspended",
        "assign-job",
        "resume-thread",
        "close-thread-handle",
    ],
)
def test_windows_setup_failure_terminates_and_closes_every_acquired_handle(
    failure_point: str,
) -> None:
    from agentseek_api import process_supervisor as supervisor_module

    api = _FakeWin32Api(fail_at=failure_point)

    with pytest.raises(supervisor_module.ProcessSupervisionError) as captured:
        supervisor_module._WindowsChild.start(
            ["command-canary"],
            env={"SECRET": "environment-canary"},
            cwd=None,
            api=api,
        )

    names = [name for name, _value in api.events]
    assert "job-handle" not in str(captured.value)
    assert "setup-canary" not in str(captured.value)
    assert "close-job-handle" in names
    if failure_point in {
        "assign-job",
        "resume-thread",
        "close-thread-handle",
    }:
        assert "close-process-handle" in names
        assert "close-thread-handle" in names
    if failure_point == "assign-job":
        assert "terminate-process" in names
    if failure_point in {"resume-thread", "close-thread-handle"}:
        assert "terminate-job" in names


def test_windows_interrupt_timeout_terminates_job_and_closes_handles() -> None:
    from agentseek_api import process_supervisor as supervisor_module

    api = _FakeWin32Api(job_empty_results=[False, True])
    child = supervisor_module._WindowsChild.start(
        ["child"],
        env={},
        cwd=None,
        api=api,
    )

    child.forward_and_reap(signal.SIGINT, timeout=5.0)
    child.close()

    names = [name for name, _value in api.events]
    assert "ctrl-break" in names
    assert "terminate-job" in names
    assert names[-2:] == ["close-process-handle", "close-job-handle"]
    wait_timeouts = [value[1] for name, value in api.events if name == "wait-process"]
    assert wait_timeouts
    assert all(timeout is not None for timeout in wait_timeouts)


def test_windows_unsupported_ctrl_break_falls_back_to_job_termination() -> None:
    from agentseek_api import process_supervisor as supervisor_module

    api = _FakeWin32Api(
        fail_at="ctrl-break",
        job_empty_results=[True, True],
    )
    child = supervisor_module._WindowsChild.start(
        ["child"],
        env={},
        cwd=None,
        api=api,
    )

    child.forward_signal(signal.SIGINT)
    api.failed = False
    child.forward_and_reap(signal.SIGINT, timeout=5.0)
    child.close()

    names = [name for name, _value in api.events]
    assert names.count("ctrl-break") == 2
    assert names.count("terminate-job") == 2
    assert names[-2:] == ["close-process-handle", "close-job-handle"]


def test_windows_ctrl_break_fallback_preserves_default_runner_exit_130(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentseek_api import cli as cli_module
    from agentseek_api import process_supervisor as supervisor_module

    api = _FakeWin32Api(
        fail_at="ctrl-break",
        job_empty_results=[True, True],
        wait_process_results=[KeyboardInterrupt(), True],
    )
    child = supervisor_module._WindowsChild.start(
        ["command-canary"],
        env={"SECRET": "environment-canary"},
        cwd=None,
        api=api,
    )
    foreground = supervisor_module.ForegroundChildSupervisor(child)
    monkeypatch.setattr(
        cli_module.ForegroundChildSupervisor,
        "start",
        lambda command, *, env, cwd: foreground,
    )

    assert (
        cli_module._default_runner(
            ["command-canary"],
            env={"SECRET": "environment-canary"},
            cwd=None,
        )
        == 130
    )

    names = [name for name, _value in api.events]
    assert "ctrl-break" in names
    assert "terminate-job" in names
    assert all(
        value[1] is not None for name, value in api.events if name == "wait-process"
    )
    assert names[-2:] == ["close-process-handle", "close-job-handle"]


def test_windows_normal_wait_polls_with_finite_intervals() -> None:
    from agentseek_api import process_supervisor as supervisor_module

    api = _FakeWin32Api(
        exit_code=47,
        wait_process_results=[False, False, True],
    )
    child = supervisor_module._WindowsChild.start(
        ["child"],
        env={},
        cwd=None,
        api=api,
    )

    assert child.wait() == 47
    wait_timeouts = [value[1] for name, value in api.events if name == "wait-process"]
    assert len(wait_timeouts) == 3
    assert all(timeout is not None and 0 < timeout <= 0.1 for timeout in wait_timeouts)
    child.close()


def test_windows_normal_return_terminates_remaining_job_members() -> None:
    from agentseek_api import process_supervisor as supervisor_module

    api = _FakeWin32Api(exit_code=31, job_empty_results=[False, True])
    child = supervisor_module._WindowsChild.start(
        ["child"],
        env={},
        cwd=None,
        api=api,
    )

    assert child.wait() == 31
    child.close_remaining_tree(timeout=5.0)
    child.close()

    names = [name for name, _value in api.events]
    assert names.count("terminate-job") == 1
    assert names.count("wait-job-empty") == 2


def test_windows_native_cleanup_failure_still_closes_every_handle() -> None:
    from agentseek_api import process_supervisor as supervisor_module

    api = _FakeWin32Api(
        fail_at="terminate-job",
        job_empty_results=[False],
    )
    child = supervisor_module._WindowsChild.start(
        ["child"],
        env={},
        cwd=None,
        api=api,
    )

    with pytest.raises(supervisor_module.ProcessSupervisionError):
        try:
            child.ensure_closed(timeout=5.0)
        finally:
            child.close()

    names = [name for name, _value in api.events]
    assert names[-2:] == ["close-process-handle", "close-job-handle"]
