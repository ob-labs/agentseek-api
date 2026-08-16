from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _use_native_interrupt_termination() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.SIG_DFL)


def _block() -> int:
    _use_native_interrupt_termination()
    while True:
        time.sleep(60)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_path", nargs="?")
    parser.add_argument("--grandchild", action="store_true")
    parser.add_argument("--parent-exit", type=int)
    parser.add_argument("--output-marker")
    args = parser.parse_args()
    if args.grandchild:
        return _block()
    if args.result_path is None:
        parser.error("result_path is required")

    _use_native_interrupt_termination()
    if args.output_marker is not None:
        print(args.output_marker, flush=True)
    blocked_signals: list[int] = []
    if hasattr(signal, "pthread_sigmask"):
        current_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        blocked_signals = sorted(
            int(signum)
            for signum in (signal.SIGINT, signal.SIGTERM)
            if signum in current_mask
        )
    grandchild = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--grandchild"]
    )
    Path(args.result_path).write_text(
        json.dumps(
            {
                "parent": os.getpid(),
                "grandchild": grandchild.pid,
                "blocked_signals": blocked_signals,
            }
        ),
        encoding="utf-8",
    )
    if args.parent_exit is not None:
        os._exit(args.parent_exit)
    return _block()


if __name__ == "__main__":
    raise SystemExit(main())
