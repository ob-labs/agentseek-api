import json
import os
from pathlib import Path
import subprocess
import sys


def test_benchmark_compares_wire_snapshots_after_memory_clear(tmp_path):
    root = Path(__file__).resolve().parents[2]
    report = tmp_path / "benchmark.json"
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_stream_persistence.py", "--backend", "sqlite",
         "--messages", "2", "--output", str(report)],
        cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checks = json.loads(report.read_text())["checks"]
    assert checks["output_matches_expected"] is True
    assert checks["thread_events_replayed_after_memory_clear"] > 0
    assert checks["run_events_replayed_after_memory_clear"] > 0
    assert checks["run_stream_replays_terminal_event_after_memory_clear"] is True
