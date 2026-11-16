<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
# CPython GIL attribution

This diagnostic runs the existing synchronous Python `Index.search` with an
isolated **CPython 3.9.19** interpreter. It does not replace Conda/system Python,
change Orion's public API, or change SIMD, prefetch, search, or GIL scheduling.
It is not a production interpreter build or a new latency leaderboard.

## Recorded boundaries

`patch_cpython.py` inserts probes in `Python/ceval_gil.h` and adds a private
`_gilprobe` built-in. `trace.h` documents the 4-u64 event record. Each worker
preallocates a bounded TLS buffer before timing. Hot-path events do not allocate,
write files, take a shared logging lock, or create Python objects. Overflow is
counted; the runner rejects any trial with dropped records. `finish()` disables
tracing and frees the buffer on its owning thread; callers must use `finally`.

Events distinguish entry mutex acquisition, condition waiting, drop requests,
GIL acquisition, logical release/notification, and forced handoff waiting.
`analyze.py` provides the event dictionary and joins requests to the existing
six `search-timing` columns. Absolute C clocks use `CLOCK_MONOTONIC` across all
threads. Rust `Instant` durations are relative; their epochs are never mixed.

Interpretation boundaries:

- `cond_wait_begin → cond_wait_end` includes condition waiting, wake scheduling,
  and reacquiring the internal pthread mutex. It is **not pure sleep time**.
- Notification does not identify its recipient. A notification-to-next-acquire
  interval cannot be labeled that thread's OS wakeup latency.
- GIL ownership is wall time, including any time the owner was descheduled.
- `take_enter → take_exit` is internal GIL acquisition. The Rust return window
  additionally contains PyO3/guard work around it.
- Requests with additional interpreter-driven acquisitions are retained raw but
  excluded from single-acquisition attribution; exclusion counts are reported.
- Pending-query counts omit interpreter-driven acquisitions and are lower bounds.
- On this macOS build, recorded C timestamps have a measured 1 µs quantum.
  Equal-time attempts cannot be ordered; zero-width intervals are excluded from
  pending counts. Sub-µs mean values are aggregate estimates, not per-event precision.
- CPython's own mutex critical section contains our clocks. Tracing can change
  contention. Always report the same-interpreter overhead controls.

## Isolated build on this macOS test host

Use a fresh scratch directory. Download the official CPython 3.9.19 source tarball
from `https://www.python.org/ftp/python/3.9.19/Python-3.9.19.tgz`, record its SHA-256,
and extract it there. Then:

```bash
python3 patch_cpython.py /path/to/Python-3.9.19
cd /path/to/Python-3.9.19
./configure --prefix=/path/to/scratch/install --without-ensurepip \
  --with-system-ffi CPPFLAGS="-I$CONDA_PREFIX/include" \
  LDFLAGS="-L$CONDA_PREFIX/lib"
make -j8 LIBS='-L/opt/homebrew/opt/gettext/lib -lintl -ldl -framework CoreFoundation'
```

Run `python.exe` directly from the build tree; installation is unnecessary.
These macOS linker flags are host-specific. The diagnostic build has no PGO and
must not be compared directly against Conda's optimized Python for speedups.
On this host, unavailable optional modules were resolved read-only from the
matching Conda 3.9.19 `lib-dynload`; NumPy 1.26.4 came from that same environment.
This is only an isolated local diagnostic, not a redistributable interpreter.

Build `orion-ann-python` with `--profile python --features search-timing`, then
copy its Python facade and native library into an isolated package directory.
Keep the normal installation unchanged. The native source itself needs no new
instrumentation; the six existing timing fields remain unchanged.

```bash
export PYTHONPATH="/path/to/package:$CONDA_PREFIX/lib/python3.9/site-packages:$CONDA_PREFIX/lib/python3.9/lib-dynload"
/path/to/Python-3.9.19/python.exe test_probe.py -v
/path/to/Python-3.9.19/python.exe test_analysis.py -v
/path/to/Python-3.9.19/python.exe run.py \
  --repo /path/to/orion --package /path/to/package \
  --inputs /path/to/experiment/2026-11-15 \
  --output /path/to/experiment/YYYY-MM-DD/gil-internals-probe
/path/to/Python-3.9.19/python.exe analyze.py /path/to/results
```

The current runner deliberately reuses the SIFT1M/GloVe25 snapshots and dataset
descriptors from the existing November 15 experiments. It refuses an existing
output directory. It does not silently build missing caches. Every query's IDs
and counts are checked against batch output before measurements.

Each L uses 1/8 callers and three rounds, with >=10,000 requests and >=1 second,
warmup and a 1-second cooldown. Three rotating controls share the same binary:

1. `bare`: existing Rust clocks, without per-request C markers or C trace events.
2. `marked`: same request markers as tracing, but internal C trace disabled.
3. `traced`: bounded C events plus the same Rust clocks and markers.

These are closed-loop measurements, not equal-arrival-rate service tests.
Result destruction and driver bookkeeping are outside per-request latency but
inside throughput timing. GIL traces include them between tagged requests.
Raw arrays, source/archive hashes, interpreter flags, summary and observer cost
must be saved together. Do not promote these figures into the public README.
