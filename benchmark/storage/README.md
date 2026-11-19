# Bounded-memory storage comparison

This standalone, unpublished benchmark reuses existing Orion graphs and sidecars.
It does not alter runtime placement policy or production SIMD/prefetch scheduling.

## Arms

| Arm | Graph access | Base and sidecars |
|---|---|---|
| `mmap` | Existing synchronous Orion search over a mapped PhasedGraph | Private mappings |
| `mmap-driver` | Suspendable SearchDriver over a mapped ReadRegion | Private mappings |
| `ssd` | SearchDriver + 4 KiB bounded page cache + FileReader | Private mappings |

The third arm currently uses buffered positional reads on Tokio's blocking pool.
It is **not io_uring or O_DIRECT**, and only graph access is explicit asynchronous
I/O. Vectors and sidecars still fault through mmap. The extra mmap-driver arm
separates driver/scratch/record-validation overhead from file-read overhead.
All backing copies are created within the measured job's memory cgroup; no
external warm inode is used as the graph read target. No host cache drop occurs.
Initialization copies, graph validation, calibration and the full-query validation/
warmup pass are excluded from search timing. Completed private output windows are synchronously written back and submitted for
file-page reclamation every 32 MiB on Linux and macOS. Linux uses `msync`,
`MADV_DONTNEED` and `POSIX_FADV_DONTNEED`; macOS uses synchronous `MS_INVALIDATE`.
The public runtime keeps its existing cache policy unless this construction
option is explicitly enabled. Advice failures abort preparation rather than
silently claiming a cold cache. Successful advice does not guarantee eviction:
pages can remain cached or refault, and Lustre server caches are outside our control.
The existing validation/warmup pass still precedes measurement; no eviction calls
are inserted into the search loop. Source dataset/cache files are never invalidated. This is a warmed, memory-pressure workload, not a cold-device
or physical-SSD-bandwidth measurement. On GLOBALFS the storage is Lustre.

## Protocol

- SIFT1M: none / L2U8 / exact-f32. GIST1M: JL / L2-KT / exact-f32.
- Existing partitioned graphs are reused. No graph construction is scheduled.
- One shared calibration per dataset, 100 queries, k=10, window=3, maximum tested L.
- SIFT L=48,128; GIST L=128,512. These are search effort points, not guaranteed recall targets.
- 1 and 8 closed-loop concurrent requests; all helper threads share 8 allowed CPUs.
- Three rotated arm orders; each L runs complete query passes for at least 30 s.
- Strict ID recall@10 and ordered-neighbor SHA-256 must match across arms.
- Request latency includes the search future's I/O waits, excluding dataset setup,
  full-query warmup and Python orchestration. QPS includes executor scheduling between requests.
- Tail quantiles with fewer than 10,000 samples are marked exploratory.

The common resident estimate counts padded base, graph payload, sidecar payload,
node counters, queries, ground truth and an explicit 256 MiB runtime allowance.
The requested cgroup limit is half or one-quarter of that estimate, rounded up to
MiB. This is a process/job memory limit including charged file cache, not merely
an RSS cap. The SSD cache occupies min(128 MiB, limit/8) **within** that limit.
`memory.max` or v1 `memory.limit_in_bytes` is checked through visible ancestors.
Unlimited/mismatched limits or unverified active swap abort the trial. Slurm
`--mem` alone is never accepted as evidence that a limit was enforced.

Counters, memory pressure, peak usage, process RSS and filesystem type are saved.
Lustre I/O may not appear in local block `io.stat`; cache misses are logical page
loads, not physical device operations. Other tenants and server-side caching can
still affect results. Controlled limits do not guarantee isolated storage hardware.

## Local validation

```bash
cargo +1.92 test --release --manifest-path benchmark/storage/Cargo.toml
cargo +1.92 clippy --manifest-path benchmark/storage/Cargo.toml --all-targets -- -D warnings
python3 -m unittest discover -s benchmark/storage -p 'test_*.py' -v
```

## Remote kit

The staged kit contains `bin/`, `data/`, scripts, and `checksums.json`.
From its root:

```bash
python3 submit.py                 # inspect memory estimates and planned commands
python3 submit.py --submit        # submit preparation + four sequential jobs
```

Preparation verifies files, runs tiny cross-arm consistency checks and calibrates
once. Failed preparation blocks all dependent measurements. Trials only read the
packaged dataset/cache inputs and delete their own RAII-managed temporary backing
files on normal exit. Forced job termination may leave named experimental graph
copies under the run's `work/`; no source cache is modified or deleted.
Reports are under `runs/<timestamp>-storage/<dataset>-one-over-<divisor>/`.
`summary.md` and `summary.json` update after each completed arm; `state.json` records
failures and their logs. No script automatically starts Orion production jobs.
