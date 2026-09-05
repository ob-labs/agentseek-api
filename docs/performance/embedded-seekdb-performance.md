# Embedded seekdb performance report

This report measures API completion time, persistence costs, and checkpoint
and event-replay correctness for sequential agent workloads using native
embedded seekdb. Measurements were collected on **2026-09-05** with no
injected database latency.

Across 24 requests on the evaluated runtime, median completion times ranged
from **211 to 243 ms**. The slowest request completed in **439 ms**. All
checkpoint, replay, and configured performance checks passed.

## Test configuration

The application used its normal `SEEKDB_EMBED=true` configuration: SQLite
for stream events and metadata, and native embedded seekdb for completed-run
checkpoints, LangGraph checkpoints, and the store. Tests called the real
FastAPI `/threads/{thread_id}/runs/wait` endpoint through the production
inline executor and a deterministic one-node LangGraph.

| Setting | Value |
|---|---|
| Embedded library | `pylibseekdb==1.4.0` |
| Native database version | `5.7.25-OceanBase seekdb-v1.4.0.0`, verified with `SELECT version()` |
| Local environment | macOS 26.6.2, ARM64, Python 3.13.2 |
| CI environment | Linux x86_64, glibc 2.39, Python 3.12 |
| Dependencies | Repository lockfile with the `embedded` extra |
| Execution | Sequential requests; no model-provider calls |
| Local samples | Five trials per workload and revision |
| CI samples | Three trials per workload on the evaluated runtime |

Each workload/revision session initialized a fresh temporary database. Every
trial used a new thread within that session. The history workload populated
its thread before the measured request. The local revision comparison used
the same script and Python environment for both revisions.

| Role | Revision |
|---|---|
| Reference runtime | `4ec7973f7f28e965c41da5011752e8df43719a24` |
| Evaluated runtime | `a68291984f22f5bfb675de91973e205f85b0d6d6` |
| Benchmark and CI configuration | `986da1e1478ec94bc4aea8872b62f9eb520b9aff` |

The benchmark configuration revision contains the same runtime code as the
evaluated revision. There were 15 local reference requests, 15 local evaluated
requests, and nine evaluated requests in CI.

## Workloads and metrics

| Workload | Prior thread events | Messages produced | Thread events after completion |
|---|---:|---:|---:|
| Fresh thread | 0 | 38 | 272 |
| Existing history | 272 | 0 | 278 |
| Larger stream | 0 | 128 | 902 |

- **API completion:** elapsed time for the measured `/runs/wait` request.
- **Framework time:** API completion minus time spent inside the graph node.
- **Finalization:** time from `execute_run` returning until the API response
  completes. The completed-run checkpoint has already been saved; this
  interval includes final stream persistence and the remaining completion-poll
  delay.
- **Metadata commits:** transactions committed to the metadata database during
  the measured request, including lifecycle and status writes.
- **Snapshot SELECTs:** queries issued by the final thread-history snapshot pass.

Application initialization is timed separately and excluded from request
metrics. It took 1.26–1.29 seconds locally and 1.74–1.95 seconds in CI for the
evaluated runtime.

## API performance

| Workload | macOS median | macOS maximum | Linux CI median | Linux CI maximum |
|---|---:|---:|---:|---:|
| Fresh thread | 211 ms | 219 ms | 229 ms | 237 ms |
| Existing history | 212 ms | 213 ms | 224 ms | 224 ms |
| Larger stream | 214 ms | 222 ms | 243 ms | 439 ms |

Finalization maxima were 187 / 197 / 155 ms locally and 157 / 184 / 195 ms in
CI, in workload order. All evaluated API and finalization durations were below
1 second and 250 ms respectively.

The `/runs/wait` endpoint polls completion every 200 ms. This accounts for
much of the response-time floor in these short workloads. Finalization is a
client-visible completion metric, not a measurement of SQL drain time alone.

## Persistence costs and revision comparison

The following comparison uses the five local trials per workload/revision.

| Workload | Reference median API | Evaluated median API | Metadata commits, reference / evaluated | Snapshot SELECTs, reference / evaluated |
|---|---:|---:|---:|---:|
| Fresh thread | 411 ms | 211 ms | 315 / 8 | 271 / 1 |
| Existing history | 213 ms | 212 ms | 11 / 6 | 277 / 1 |
| Larger stream | 1,427 ms | 214 ms | 1,035 / 14 | 901 / 2 |

The commit and snapshot SELECT counts in the table were consistent across
all trials of each workload and matched between macOS and Linux for the
evaluated runtime. The harness reports native seekdb checkpoint operations
separately from metadata operations.

Median final snapshot durations on macOS were 1.0 / 1.3 / 3.1 ms. The history
workload's API timing remains nearly unchanged because both revisions finish
within the completion-poll interval, even though their query counts differ.

## Correctness and acceptance criteria

Every measured request verified:

- Exact expected output content and saved completed-run checkpoint contents.
- LangGraph checkpoint contents, the actual native saver, database version,
  and physical checkpoint rows.
- Exact sequence and payload equality for all thread events and run SSE
  events replayed after clearing the in-memory brokers.
- Matching output digests, checkpoint checks, and replay counts across the
  local reference and evaluated revisions.

| Check | Local limit | Shared CI runner limit |
|---|---:|---:|
| Maximum API completion | 1 s | 3 s |
| Maximum finalization | 250 ms | 1 s |
| Metadata commits: fresh / history / larger | 16 / 16 / 24 | 16 / 16 / 24 |
| Snapshot SELECTs: fresh / history / larger | 1 / 1 / 2 | 1 / 1 / 2 |

CI timing limits allow for shared-runner variability. SQL operation budgets
provide a performance check independent of machine speed. As a validation of
the checks themselves, the reference runtime was run with the fresh-workload
SQL budgets and exited with failures for both excessive commits and SELECTs.

The [Linux embedded test job](https://github.com/ob-labs/agentseek-api/actions/runs/33961392416/job/101293768899)
passed all three workloads. Its workflow run contains the
`embedded-seekdb-performance` artifact with per-trial JSON measurements.

## Reproduction

From the repository root, install the locked optional backend and run the
three local workloads:

```sh
uv sync --frozen --extra embedded

PYTHONPATH=src uv run python scripts/benchmark_stream_persistence.py \
  --backend embedded --repeat 5 --messages 38 \
  --max-api-seconds 1 --max-finalization-seconds 0.25 \
  --max-metadata-commits 16 --max-snapshot-selects 1 \
  --output /tmp/embedded-fresh.json

PYTHONPATH=src uv run python scripts/benchmark_stream_persistence.py \
  --backend embedded --repeat 5 --messages 0 --history 38 \
  --max-api-seconds 1 --max-finalization-seconds 0.25 \
  --max-metadata-commits 16 --max-snapshot-selects 1 \
  --output /tmp/embedded-history.json

PYTHONPATH=src uv run python scripts/benchmark_stream_persistence.py \
  --backend embedded --repeat 5 --messages 128 \
  --max-api-seconds 1 --max-finalization-seconds 0.25 \
  --max-metadata-commits 24 --max-snapshot-selects 2 \
  --output /tmp/embedded-larger.json
```

To evaluate another revision, point `PYTHONPATH` at its archived `src`
directory while keeping the script and environment the same. Omit limits
when collecting an unconstrained reference measurement. Reports include the
source path, backend identity, individual samples, SQL counts, summary
statistics, correctness checks, and any exceeded limits.

## Measurement scope

These results describe short, sequential, deterministic workloads on the
listed environments. They exclude provider latency and application startup
from request timing and do not measure concurrent load or crash recovery.
The small sample sets provide observed medians and maxima, not production
tail-latency guarantees.
