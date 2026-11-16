# USearch binding control

This isolated benchmark investigates Python request-tail latency. It does not
replace the stock USearch arm or modify a running baseline experiment.

Three arms read the same f32 USearch index:

| Arm | Binding | GIL during native search |
|---|---|---|
| `python` | Installed USearch 2.23.0 Python API, `threads=1` | Held |
| `native-gil` | Private C ABI through `ctypes.PyDLL` | Held |
| `native-nogil` | Identical C ABI through `ctypes.CDLL` | Released |

The native adapter reserves eight contexts before search. Each request leases an
exclusive scratch slot and copies its results before returning the slot. A
lifecycle gate prevents destruction during active calls. There are no internal
search workers; concurrency is provided by the same 1/8 Python request clients.
Configuration changes are outside measurement and serialized against searches.
The installed SimSIMD library is shared; actual distance backends must agree.

## Build and test

Use a clean USearch v2.23.0 checkout at the commit pinned in `build.py`, with its
`fp16` and `simsimd` submodules initialized. From the repository root:

```sh
python -m benchmark.latency_baselines.usearch_control.build \
  --source /path/to/usearch --output /path/to/native
USEARCH_CONTROL_LIBRARY=/path/to/native/liblatency_usearch.dylib \
  python -m unittest discover -s benchmark/latency_baselines/usearch_control \
  -t . -p 'test_*.py'
```

Linux uses `.so`. The control is pinned to the installed 2.23.0 wheel and records
source, library and SimSIMD hashes. Tests cover three metrics, short/full rows,
serial/concurrent IDs and distances, scratch contention, invalid inputs,
lifecycle, GIL release, deferred execution and a tiny complete harness run.

## Deferred execution

Run `python -m latency_baselines.usearch_control --plan /absolute/plan.json`
from the benchmark directory (or a frozen copy). The plan records the preceding
process PID, start time and command, all expected result directories, and jobs.
The process must have exited **and** every expected result must be terminal
before any dataset is loaded. Missing, failed or unfinished jobs stop the
control for inspection. Incomplete measurements and explicitly unsupported or
missing engines are terminal. PID reuse cannot falsely extend the wait.

Each job specifies `dataset`, `library`, `efforts`, `k` and `protocol`, using the
existing baseline dataset schema. Owned temporary graphs are built once and
shared across all arms/rounds, then deleted. Historical graphs are untouched.
Performance results from a rebuilt graph must not be treated as an exact paired
comparison against an older graph.

The scheduled diagnostic uses SIFT (ef=64), DEEP10M (128), and GloVe100 (1024),
where the earlier run showed tails. All use k=10, 1/8 clients, three rounds,
1 second cooldown, warmup, and the unchanged minimum 10,000 requests + 1 second
protocol (30-second ceiling). Variant order rotates between rounds. Before
measurement, all query IDs and scores are compared; concurrent IDs must match.

Outputs retain raw nanosecond samples and SHA-256, pooled P50/P95/P99, median
round QPS, recall, actual backend and completion flags. Timing remains Python
request-boundary wall time, including allocations and GIL reacquisition—not
pure native-kernel latency. The `native-gil` vs `native-nogil` comparison isolates
GIL policy within the new adapter; stock vs native also changes binding and
context-management overhead. The tiny test's timings are not performance data.
