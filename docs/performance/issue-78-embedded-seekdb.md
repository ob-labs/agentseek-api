# Issue #78: native embedded seekdb validation

The fix meets the local acceptance targets on all 15 patched requests: API
completion under 1 second and client-visible finalization under 250 ms. The
slowest API request took **222 ms**. No database latency was injected.

## Environment and method

- macOS 26.6.2, ARM64; Python 3.13.2; frozen repository dependencies.
- `pylibseekdb==1.4.0`; native `SELECT version()` returned
  `5.7.25-OceanBase seekdb-v1.4.0.0`.
- Normal `SEEKDB_EMBED=true` configuration: SQLite stream/metadata storage,
  native embedded seekdb run checkpoints, LangGraph checkpoints, and store.
- Baseline: `4ec7973f7f28e965c41da5011752e8df43719a24`.
  Runtime fix: `a68291984f22f5bfb675de91973e205f85b0d6d6`.
- Same benchmark script and Python environment for both revisions. Each
  workload/revision ran five requests sequentially, each on a new thread,
  against a fresh temporary embedded database. The history workload seeded
  its own thread before timing. Thirty measured requests in total.
- Real FastAPI `/threads/{thread_id}/runs/wait`, production inline executor,
  and a deterministic one-node LangGraph. No model-provider calls and no
  replacement executor, persistence implementation, or checkpointer.

Application initialization is excluded from request timing and recorded
separately: 1.26–1.29 seconds for the patched runs. The deterministic node's
own computation took less than 1 ms per request, so API timing primarily
measures framework work and the completion poll.

## Results

| Workload | Thread events after completion | Baseline median API | Fixed median API | Fixed maximum API |
|---|---:|---:|---:|---:|
| Fresh thread, 38 messages | 272 | 411 ms | **211 ms** | 219 ms |
| 272 prior events, then an empty-message run | 278 | 213 ms | **212 ms** | 213 ms |
| Fresh thread, 128 messages | 902 | 1,427 ms | **214 ms** | 222 ms |

| Workload | Metadata commits before → after | Final snapshot SELECTs before → after |
|---|---:|---:|
| Fresh, 38 messages | 315 → **8** | 271 → **1** |
| Existing history | 11 → **6** | 277 → **1** |
| Fresh, 128 messages | 1,035 → **14** | 901 → **2** |

Counts were consistent across all five trials of each workload. They include
metadata lifecycle/status writes; native seekdb checkpoint operations are
reported separately by the harness. Fewer commits do not mean fewer saved
events: all expected payloads and sequences were verified.

The existing `/runs/wait` endpoint polls completion every 200 ms. That largely
explains the approximately 211–214 ms floor and why the local history case
has nearly unchanged API timing despite eliminating 276 snapshot queries.
`finalization_seconds` includes this residual polling delay; it is not a SQL
drain timer. Patched finalization maxima were 187 / 197 / 155 ms. Median final
snapshot passes took 1.0 / 1.3 / 3.1 ms respectively.

## Correctness and regression checks

Every request checked exact expected output content, the completed-run
checkpoint payload, and graph checkpoint contents. Embedded mode additionally
checks the actual saver class, native version query, and physical checkpoint
rows. After clearing the brokers, all thread events and every run SSE event
were compared for exact sequence/payload equality. Output digests, checkpoint
checks, and replay counts also matched across revisions.

The existing Embedded seekdb Smoke CI job now runs three trials of each
workload and uploads `embedded-seekdb-performance` JSON artifacts. Shared
runners have a 3-second API / 1-second finalization limit; deterministic
commit budgets (16 / 16 / 24) and snapshot SELECT budgets (1 / 1 / 2) catch
regressions independently of machine speed. The original implementation was
run with the new fresh-workload budgets and correctly exited with failures
for both excessive commits and excessive snapshot SELECTs.

## Reproduce

Install the locked optional backend and run the strict local fresh-thread
check from the repository root:

```sh
uv sync --frozen --extra embedded
PYTHONPATH=src uv run python scripts/benchmark_stream_persistence.py \
  --backend embedded --repeat 5 --messages 38 \
  --max-api-seconds 1 --max-finalization-seconds 0.25 \
  --max-metadata-commits 16 --max-snapshot-selects 1 \
  --output /tmp/embedded-fresh.json
```

For history, use `--messages 0 --history 38`. For the larger stream, use
`--messages 128 --max-metadata-commits 24 --max-snapshot-selects 2`.
To compare the baseline, point `PYTHONPATH` at that revision's archived `src`
directory and omit performance limits; keep the script and environment the
same. The report records the source path, backend identity, every sample,
SQL counts, summary statistics, and correctness checks.

These measurements validate local embedded performance, not the reporter's
unavailable deployment. They exclude model latency and application startup.
The existing documented limitation still applies: an abrupt process failure
can lose unflushed stream events; normal execution exit drains the buffer.
