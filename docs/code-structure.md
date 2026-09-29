# Code Structure

The standalone executable and shared runtime now live in `orion-cli`:

```text
orion-cli/src/
  main.rs                  # One-pass search and preparation; JSONL output
  lib.rs                   # Shared runtime API, independent of benchmark
  config.rs                # Embedded YAML presets and shared settings
  cascade.rs               # Stage selection and f32 dispatch
  parlayann_bridge.rs      # STAG import
  cli/
    config.rs / plan.rs    # Resolve options into one validated execution plan
    data.rs               # Base ownership and replayable experiment inputs
    query.rs              # QuerySource + Read-based decoding + optional GT
    execution.rs          # SearchBackend implementations
    index.rs / cache.rs   # Build/import and cache provenance
    resources.rs          # Component-level memory report and budget checks
benchmark/src/bin/
  orion_sweep.rs           # orion-sweep: repeated L/trial experiments
```

The benchmark's configuration/cascade/utility modules re-export shared runtime
code; they no longer own duplicate CLI implementations. The older tree below
also describes the engine and remaining benchmark tools.


[← Back to main README](../README.md)


```
orion/
  src/
    model/
      phased_graph.rs               # PhasedGraph: AlignedBoxWithSlice slab + 60%-by-distance
                                    #   (origin + extras) partition + write queue
      dataset/
        quantized_dataset.rs        # QuantizedDataset<Q,N> (L2U8 / L2U16 / MipsI8 / MipsI16) +
                                    #   AlignedBoxWithSlice 32B-stride storage + sidecar load/build
        l2_kt_dataset.rs            # L2 kernel-trick: i8 base + per-vert ‖x_i8‖² for
                                    #   `‖q−x‖² = ‖q‖² + ‖x‖² − 2·⟨q,x⟩` (sdot-friendly)
        jl_sparse_dataset.rs        # JL Sparse 1024-bit signature — NZ=9 sparse projection
                                    #   with balanced per-dim coverage (Fisher-Yates over multiset)
        jl_hadamard_dataset.rs      # JL Hadamard 1024-bit — HDHDHD sign-pack, dense per-bit
        rabitq_dataset.rs           # RaBitQ B=1 sign-bit base (rotation + signs)
        rabitq_b4_dataset.rs        # RaBitQ B=4 with per-vertex correction
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
        in_mem_search.rs            # **Unified cascade beam loop** — search_unified /
                                    #   search_batch_unified, generic over (P, A, R) trait
                                    #   triple; one body monomorphises per recipe
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

vector/
  src/
    distance_stream.rs              # DistanceStream<K,N>: inline-prfm address resolve,
                                    #   flat 4-way (lpv, stride) dispatch, pldl1strm hint,
                                    #   prologue + per-iter drip + **sink-time burst (LA=12)**
                                    #   for long-range L1 lookahead — +20-27% QPS on GIST L2-KT
    l2_neon_distance.rs             # L2U8Distance, L2F32Distance, L2U16Distance kernels (NEON)
    ip_neon_distance.rs             # IpI8Distance, IpI16Distance, IpF32Distance kernels (NEON)
    distance_fn.rs                  # DistanceFn trait — kernel ABI for DistanceStream,
                                    #   JLHammingDistance (XOR+popcount for bit-packed signatures)

diskann/
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
      orion.rs             # **Production driver** — composable cascade: --prefilter --admission --rerank
      diskann_sweep.rs              # Microsoft DiskANN L-sweep (in-process via the `diskann` core crate)
                                    #   — third engine for the 3-engine head-to-head
    runner/
      cascade.rs                    # **Cascade dispatch layer** — PrefilterChoice / AdmissionChoice
                                    #   / RerankChoice enums + build_*() factories +
                                    #   pin_cascade() mlock helper + search_compose /
                                    #   search_batch_compose entry points
      parlayann_bridge.rs           # Load PA `.staged v3` exports → (local, remote, extra) per-node
      orion_runner.rs      # Boxed AlgorithmRunner; reads cascade triple from
                                    #   sweep.yaml's per-dataset `orion.{prefilter,admission,rerank}`
      orion_ads_runner.rs  # ADSampling variant — rotates the dataset, dispatches through
                                    #   the AdsF32Admission cascade
      diskann_runner.rs / diskann_ads_runner.rs  # In-memory DiskANN baselines
    config.rs                       # sweep.yaml parser — `cfg.datasets.<ds>.staged.{alpha, R,
                                    #   L_build, max_extra, window_size, prefilter, admission,
                                    #   rerank}` typed config
  configs/
    sweep.yaml                      # **Single source of truth** for per-dataset build params +
                                    #   cascade triple (replaces the legacy 4-way `metric:` enum)
  scripts/
    prepare_parlayann_data.sh                 # Build PA `.staged` exports (one-time, per dataset)
    sweep_orion_vs_diskann_vs_parlayann.sh   # 3-engine head-to-head: Orion + MS Vamana + PA Vamana,
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
