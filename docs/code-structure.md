<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# <img src="../orion-core/assets/orion-astraeus-heading.svg" alt="" width="72" align="middle" /> Code Structure

The runtime is grouped by responsibility and shared directly by CLI, Python and
benchmark callers. `orion-cli` is binary-only: it has no library or forwarding
modules. CLI syntax and conversion to `IndexOptions` live in `orion-cli`.
YAML, dataset shortcuts and legacy experiment environment resolution live only
in the independent benchmark workspace. The runtime supplies shared validation
and execution services to all adapters.

```text
orion-runtime/src/          # Public orion-ann facade
  lib.rs                   # Small application-facing API and diagnostics exports
  index/                   # Owned resident session and dedicated Rayon pool
    base.rs                # File/array input identity, validation and normalization
    prepare/
      mod.rs               # Ordered load lifecycle; writer lease and final publication
      input.rs             # Base budget, original identity and graph selection
      source.rs            # Snapshot/supplier verification and source budget checks
      graph.rs             # Element/source dispatch into a read view
      snapshot.rs          # Verified graph and sidecar restoration
      working.rs           # Working graph load/build and provenance publication
      search.rs            # Auxiliary state and saved-calibration readiness
      identity.rs          # External ID publication and label restoration
    calibration.rs         # Per-k calibration and derived stage preparation
    query/mod.rs           # Borrowed QueryBatch, SearchOptions and shared search
    view.rs                # Resident generation, payload identities and engine
    snapshot/              # Manifest encoding, verification and publication
  config/
    options.rs             # IndexOptions aggregation and service resolution
    data.rs                # Metric and resident storage representation
    graph.rs               # Graph policy, recipe and core BuildConfig conversion
    execution.rs           # Worker/window bounds and resolved sweep record
    resources.rs           # Budget validation and GiB conversion
    cascade/               # Choices, defaults, compatibility and requirements
    plan.rs                # Base/search compatibility boundary
    resolved.rs            # Resolved run and stable graph-cache identity
  execution.rs             # Optional adapter support; no parsing or presets
  data/                    # Formats, padded readers and streaming query decoding
  graph/
    mod.rs                 # Build/import/load orchestration
    selection.rs           # Automatic or explicitly strict source selection
    cache.rs               # Working lookup records and build provenance
    persistence/mod.rs     # Graph encoding and verified decoding
    staged.rs              # STAG import and metadata validation
    parlayann.rs            # In-memory ParlayANN construction
  sidecars/                # Deduplicated requirements and working sidecar storage
  preflight/               # Memory budgets and CLI reports
  integrity.rs             # SHA-256 calculation and verified decoder streams
  storage/                 # Immutable object store and atomic publication
  search/                  # Runtime configuration adapters to core stage recipes
  benchmark.rs             # Experimental controls behind the benchmark feature

orion-parlay/
  src/lib.rs               # Safe borrowed builder handle over a private C ABI
  cpp/bridge.cpp           # f32/u8 point views, explicit schedulers and error boundary
  vendor/                  # Pinned ParlayANN/Parlaylib sources and licenses

orion-cli/
  src/main.rs              # One-pass CLI, JSONL results and calibration summary
  src/args.rs              # CLI syntax
  src/resolve.rs           # Explicit input -> shared IndexOptions
  tests/                   # CLI execution and offline-preflight integration tests

benchmark/
  configs/sweep.yaml       # Experiment-only dataset paths and sweep defaults
  src/config/             # YAML/preset parsing and benchmark CLI resolution

orion-py/
  src/lib.rs               # PyO3 ownership boundary, GIL release and per-index lock
  python/orion_ann/        # Public Index/config/result API
  tests/                   # NumPy/file/cache and CLI result consistency checks

orion-core/src/
  build/                            # All graph producers converge on the same slab
    mod.rs                          # Standalone build entry points and backend selection
    config.rs                       # Shape, metric, worker and backend capability checks
    partitions.rs                   # Validated in-memory partition and STAG construction
    parlay.rs                       # ParlayANN writes directly into the final slab
    diskann.rs                      # Rust builder adapter
    diskann_builder.rs              # Vamana graph + partition extraction
    supplier/                       # Experimental named partition sources and staged codec
  index/
    mod.rs                          # Orion ownership: base, graph, entry, sidecars, scratch
    access.rs                       # Read-only access and ownership transfer
    graph.rs                        # Borrowed, read-only GraphView
    codec.rs                        # Graph encoding/decoding; runtime publishes/verifies objects
  search/
    mod.rs                          # Execution-phase map and opt-in counters
    query/
      mod.rs                        # Shared padding and normalization helpers
      input.rs                      # Typed borrowed/owned transport; scalar declaration table
      prepared.rs                   # Per-row representations shared by cascade sessions
      numeric.rs                    # Wide mixed-type exact arithmetic
      controls.rs                   # Search controls and validation
    cascade/
      mod.rs                        # Quantized stage recipe and borrowed query plan
      plan.rs                       # Bind preparation requirements and official stage sessions
      execute.rs                    # Common query/session lifecycle and scheduling
      stage/
        prefilter/                  # Official cheap rejection stages
        admission/                  # Official candidate scoring stages
        rerank/                     # Official final ranking stages
    traversal/
      mod.rs                        # Existing beam loop and scratch-pool use
      tuning.rs                     # Traversal/prefetch constants
    filtered/
      mod.rs                        # Exact allowed-ID scan; runtime resolves labels/revisions
    output.rs                       # Original-metric scoring and validated result assembly
    adaptive/
      calibrate.rs                  # Statistics and parameter estimation
      calibration.rs                # Index calibration API and sampling
      convergence.rs                # Admission-rate sliding window
      early_exit.rs                 # Consecutive-zero-admit countdown
    diagnostics/
      metrics.rs                    # Optional counters
      analysis/                     # Retained benchmark-only neighbor/contribution analysis
  model/
    dataset/
      base/                         # Base shape/placement and coordinate/metric contracts
      native_u8.rs                  # Exact squared norms, placement and serialization
      store/                        # Lazy sidecar ownership, accessors and storage configuration
      quantized_dataset/
        scalar/                     # L2U8/L2U16/MipsI8/MipsI16
        kernel_trick.rs             # Signed codes and per-vector norms
        jl/                         # Sparse and Hadamard binary sketches
        rabitq/                     # Binary/multibit rotated codes
        pq/                         # Independent slice-based PQ and ADC stream
        exact/                      # Identity sessions borrowing the base, including native IP/cosine
    phased_graph/mod.rs             # Aligned local/remote/extra slab; no public mutable graph
    neighbor/                       # Candidate priority queues and merge primitives
    scratch.rs                      # Reused query-local visited/candidate buffers
    search_result.rs                # IDs/counts and optional original-metric scores

orion-vector/
  src/
    distance_stream.rs              # DistanceStream<K>: address resolution, bounded near cursor,
                                    #   group/burst scheduling and intentional near/far overlap
    stream_kernels/                 # Runtime ISA dispatch and inlined arithmetic backends
      compute/mod.rs                # Unrolled compute; prefetch credit per CACHE_LINE_BYTES,
                                    #   including each vector's partial final cache line
    l2_neon_distance.rs             # L2U8Distance, L2F32Distance, L2U16Distance kernels (NEON)
    ip_neon_distance.rs             # IpI8Distance, IpI16Distance, IpF32Distance kernels (NEON)
    distance_kernel.rs              # DistanceKernelBackend trait — stream kernel ABI;
                                    #   JLHammingDistance for bit-packed signatures

orion-diskann/
  src/
    model/graph/
      vertex_and_neighbors.rs       # Distance-sorted neighbor maintenance (add_sorted, neighbor_dists)
    algorithm/prune/
      prune.rs                      # Slab writes: (distance, pruned_id) per location
    index/inmem_index/
      inmem_index.rs                # extract_graph_and_candidates: returns (local, remote, extra)
                                    #   partitions sourced from the merged origin+extras pool

benchmark/
  src/
    bin/
      orion_sweep.rs                # Replay, calibration and parameter/trial sweeps via runtime
      diskann_sweep.rs              # Microsoft DiskANN L-sweep (in-process via the `diskann` core crate)
                                    #   — third engine for the 3-engine head-to-head
    runner/
      mod.rs                        # Direct runtime cascade/STAG imports and runner exports
      orion_runner.rs      # Boxed AlgorithmRunner; reads cascade triple from
                                    #   sweep.yaml's per-dataset `orion.{prefilter,admission,rerank}`
      orion_ads_runner.rs  # ADSampling variant — rotates the dataset, dispatches through
                                    #   the AdsF32Admission cascade
      diskann_runner.rs / diskann_ads_runner.rs  # In-memory DiskANN baselines
  configs/
    sweep.yaml                      # **Single source of truth** for per-dataset build params +
                                    #   cascade triple (replaces the legacy 4-way `metric:` enum)
  scripts/
    prepare_parlayann_data.sh                 # Build PA `.staged` exports (one-time, per dataset)
    sweep_orion_vs_diskann_vs_parlayann.sh    # 3-engine head-to-head: Orion + MS Vamana + PA Vamana,
                                              #   3-run × 180 s thermal-isolated
    run_build_memory_profile.sh               # α-matched build + peak-memory profile across 4 datasets
    run_thread_scaling.sh                     # QPS-vs-thread curves on SIFT 1M + GIST 1M
                                              #   (M4 Max P-core knee at T=10)
    run_sift_baseline_panel.sh                # SIFT 1M panel — HNSW + FAISS x2 + Annoy + the 3 Vamana engines
    run_ads_benchmark.sh                      # ADSampling 4-way (DiskANN, +ADS, Orion, +ADS) on $DATASET
    run_ablations.sh                          # Unified ablation driver — runs both panels per dataset:
                                              #   `--algorithms ablation` (origin/full/no-EE/no-extra → plot_ablation.py)
                                              #   `--algorithms cascade-ablation` (4-tier → plot_cascade_ablation.py)
```

## Diagnostics and browser replay

- `orion-error/`: the single public `Error`, `ErrorCode`, `ErrorReport` and `Result`
  implementation. Native and WASM frontends use the same wire report.
- `orion-core/`: search and lightweight `SearchObserver` hooks. It re-exports shared
  errors and has no dependency on the new diagnostics crate.
- `orion-diagnostics/`: schema, validation and replay. Its default `native`
  feature depends on Orion and supplies the production observer adapter.
- `orion-runtime/`: `Index::trace_search` shares normalization, calibration and
  backend selection with ordinary search.
- `orion-wasm/`: browser-facing replay, using diagnostics with default features
  disabled. It does not yet execute graph search. Windows adaptation follows
  the initial WASM functionality.

DiskANN owns its optional conversion to the shared error; the error crate never
imports DiskANN. Existing `orion_ann::Error` and `orion_ann_core::error` imports remain
valid re-exports. No duplicate native error wrapper is introduced.

## Core execution boundary

`Index` in the runtime owns application policy, validation, persistence, labels and
worker limits. The core `Orion` owns search state. Construction backends and
experimental suppliers both produce the same validated graph partitions; supplier
names are provenance, not search-time dispatch keys.

A search call binds its query plan before entering the common executor. The plan
prepares a query, opens official stage sessions and finalizes selected candidates.
The executor owns scheduling and observer lifetime; `output` owns result assembly; the beam loop
only sees prefilter/admission sessions. Native u8 uses the same cascade as f32: exact admission can skip rerank,
while quantized admission can add an exact rerank stage. Its IP/cosine output scores
retain wide integer accumulation, while exact L2 reuses queue distances. Quantized
f32 recipes reuse exact rerank scores where available and otherwise score only
selected IDs. Requesting scores never silently changes candidate selection.

Prepared queries outlive all their sessions. Identity sessions remain stack values;
quantized factories keep their existing boxed sessions. No datatype dispatch was
added to the per-neighbor loop, and the SIMD streams, prefetch schedule, partition
order and adaptive stopping rules are unchanged. The benchmark-only `analysis`
module remains under search diagnostics and is not part of production execution.

Search folders follow execution phases, not base datatypes. Ordinary requests
prepare a query, bind a cascade, traverse the graph and assemble output. Filtered
requests receive a sorted, unique internal-ID set from runtime label evaluation
and use the exact baseline in `filtered`; labels, external IDs and label revision
publication never enter the core traversal. `adaptive` and `diagnostics` are
orthogonal policy/observation modules shared where their contracts apply.

Complex modules use `directory/mod.rs` as their single entry point; child files
stay beside that entry. Cascade recipes always supply all three stage handles:
`NoPrefilter` and `NoRerank` explicitly disable optional work. The session trait owns the complete filtering operation, including shared
threshold maintenance and work accounting. `NoPrefilter` opens a zero-sized
session whose operation returns a constant outcome without reading or writing
scratch. Active kernels share one default operation and keep their existing
distance/compaction methods. Traversal invokes the same object-safe session
method in every case, then reports its outcome through the generic observer.
There is no optional session, dispatch macro or duplicated traversal body.
