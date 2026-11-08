<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# Code Structure

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
    prepare.rs             # Build or restore working graph and sidecar state
    calibration.rs         # Per-k calibration and derived stage preparation
    query.rs               # Borrowed QueryBatch, SearchOptions and shared search
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
    persistence.rs         # Graph encoding and verified decoding
    staged.rs              # STAG import and metadata validation
    parlayann.rs            # In-memory ParlayANN construction
  sidecars/                # Deduplicated requirements and working sidecar storage
  preflight/               # Memory budgets and CLI reports
  integrity.rs             # SHA-256 calculation and verified decoder streams
  storage/                 # Immutable object store and atomic publication
  search/                  # f32/native-u8 backend and official cascade dispatch
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

orion-core/
  src/
    model/
      phased_graph.rs               # PhasedGraph: AlignedBoxWithSlice slab + 60%-by-distance
                                    #   (origin + extras) partition + write queue
      dataset/
        quantized_dataset/
          mod.rs                    # Sealed sidecar contract: shape, build, compute and memory
          capabilities.rs           # Representation identities and cascade precision ordering
          residency.rs              # Borrowed resident buffers, independent of cascade position
          scalar/                   # Padded L2U8 / L2U16 / MipsI8 / MipsI16 and their specs
          kernel_trick.rs           # i8 base + per-vector norms for squared-L2 via sdot
          jl/sparse.rs              # Sparse JL binary signatures (L2 and MIPS)
          jl/hadamard.rs            # HDHDHD sign-packed JL signatures
          rabitq/binary.rs          # RaBitQ B=1 rotation/sign codes
          rabitq/multibit.rs        # RaBitQ B=4 with per-vector correction
          pq/                       # Independent slice-based PQ, ADC stream, code residency and codec
          exact/                    # Borrowed exact-distance sessions; no sidecar allocation
      neighbor/neighbor_priority_queue.rs  # NeighborPriorityQueue + pad16 buffer alignment +
                                           #   batch_merge / batch_merge_gallop
      scratch.rs                    # InMemSearchScratch: HashsetSeen (linear-probe, exact)
                                    #   + dist_buffer (cap=400) + DCC + EarlyExitChecker
    algorithm/
      search/
        mod.rs                      # **Shared metric machinery** — PerThreadMetrics +
                                    #   MetricsTable + define_metric! handles
                                    #   (VISIT_COUNT, RAW_VISIT_COUNT, NDC_I8, NDC_F32, SETUP_NS, …)
                                    #   used by every cascade stage + the unified loop
        in_mem_search.rs            # **Shared cascade/native-L2 traversal** — search_sessions
                                    #   Legacy search_unified / search_batch_unified wrappers
                                    #   are exposed only with the benchmark feature
        utils.rs                    # Search-loop tuning constants (FLUSH_INTERVAL,
                                    #   insert_route_mul / linear_merge_mul, dstream_la_*),
                                    #   AlignedQuery, SearchProfile{,Stats}, search_diag /
                                    #   search_profile diagnostics, PQ helpers
        stage/                      # **Composable cascade trait modules**
          prefilter/                #
            mod.rs                  #   PrefilterStage + PrefilterSession (object-safe)
            jl.rs                   #   JlPrefilter — JL Sparse signature filter
            jl_hadamard.rs          #   JlHadamardPrefilter — HDHDHD-encoded filter
            rabitq.rs               #   RabitqPrefilter — rotation-based sign code
          admission/                #
            mod.rs                  #   AdmissionStage + AdmissionSession (object-safe)
            l2u8.rs                 #   L2U8Admission — direct u8 squared-L2
            l2u16.rs                #   L2U16Admission — u16 squared-L2 (PA-quantize_bits=16)
            l2kt.rs                 #   L2KTAdmission — i8 sdot + kernel-trick reconstruction
            mips_i8.rs              #   MipsI8Admission — i8 sdot for unit-normalised data
            mips_i16.rs             #   MipsI16Admission — i16 IP for high-recall band
            ads_f32.rs              #   AdsF32Admission — ADSampling scaled-partial-sum
                                    #     early-abort L2 (requires rotated dataset)
          rerank/                   #
            mod.rs                  #   RerankStage trait
            f32_truth.rs            #   F32Rerank — full f32 base, L2 distance
            ip_f32_truth.rs         #   IpF32Rerank — full f32 base, IP distance, query unit-norm
            u16_truth.rs            #   U16Rerank — u16 sidecar, L2 (PA-bit-exact recipe)
        convergence.rs              # Admission-rate sliding window (DCC)
        early_exit.rs               # Consecutive-zero-admit countdown
        calibrate.rs                # Auto-calibration of (threshold, early_exit_limit)
        jl_hamming_cache.rs         # Per-query JL bitmap reuse across the beam loop
    index/
      builder.rs                    # build_diskann_index: Vamana build + partition extraction
      compressed_index.rs           # Orion: ensure_quantized_dataset_*() accessors,
                                    #   PhasedGraph construction, OnceLock-cached sidecars

orion-vector/
  src/
    distance_stream.rs              # DistanceStream<K>: address resolution, bounded near cursor,
                                    #   group/burst scheduling and intentional near/far overlap
    stream_kernels/                 # Runtime ISA dispatch and inlined arithmetic backends
      compute.rs                    # Unrolled compute; prefetch credit per CACHE_LINE_BYTES,
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
