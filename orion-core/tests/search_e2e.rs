/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! End-to-end integration tests that build a tiny synthetic Vamana
//! graph and exercise the validated cascade search entry points. The fixture
//! retains static storage for the legacy profiling scenarios below.
//!
//! Exists to lift `search/{in_mem_search, calibrate, utils}`,
//! `index/compressed_index`, and `orion-vector/distance_{fn,stream}` from
//! 0% line coverage — those files only run when a real PhasedGraph +
//! ScalarQuantizedDataset are wired together, which unit tests can't do in
//! isolation.
//!
//! Synthetic recipe: a 128-point dataset over a 32-dim grid. Each
//! vertex `i` has coordinates derived from a deterministic xorshift
//! seed offset so distances are well-distributed but reproducible.
//! With graph_degree=16 / search_list_size=48 / alpha=1.15 the
//! resulting graph has the structural variety needed to exercise the
//! beam-search branches (admission, convergence, early exit) without
//! the build itself taking longer than a few hundred ms.

#[path = "support/sidecar_file.rs"]
mod sidecar_file;
use orion_ann_core::search::cascade::stage::Prefilter;
use orion_ann_core::search::cascade::stage::admission::{
    L2KTAdmission, L2U8Admission, L2U16Admission, MipsI8Admission,
};
use orion_ann_core::search::cascade::stage::prefilter::{
    JlHadamardPrefilter, JlPrefilter, RabitqPrefilter,
};
use orion_ann_core::search::cascade::stage::rerank::{F32Rerank, IpF32Rerank, U16Rerank};
use orion_ann_core::{DynamicCascade, DynamicSearchConfig, StaticOrion, build_diskann_index};
use orion_ann_diskann::model::InmemDataset;

const DIM: usize = 32;
// 128 points (vs the earlier 64) gives Vamana enough candidates that
// the prune step always has a non-empty pruned_list — at 64 + dim=32
// the instrumented binary under `cargo llvm-cov` occasionally hit an
// "all candidates equidistant" edge case that the build rejects.
const N_POINTS: usize = 128;
const GRAPH_DEGREE: u32 = 16;
const BUILD_L: u32 = 48;
const MAX_EXTRA: usize = 16;

/// Deterministic xorshift — same seed → same dataset, so test
/// failures reproduce 1:1.
fn xrng(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

fn random_unit_float(state: &mut u64) -> f32 {
    let u = (xrng(state) >> 32) as u32;
    let unit = (u as f32) / (u32::MAX as f32);
    // Map to [-1, 1].
    unit * 2.0 - 1.0
}

/// Build the synthetic flat f32 dataset: `N_POINTS × DIM` row-major.
fn make_flat_dataset() -> Vec<f32> {
    let mut state: u64 = 0xCAFE_BABE_DEAD_BEEFu64;
    (0..N_POINTS * DIM)
        .map(|_| random_unit_float(&mut state))
        .collect()
}

/// Build a `StaticOrion<DIM>` from the synthetic dataset, mirroring
/// `benchmark/src/runner/orion_runner.rs::build_orion!`.
fn build_orion_fixture(flat: &[f32]) -> StaticOrion<DIM> {
    // This is a search test, not a concurrent-construction test. Tiny Vamana
    // graphs can stall in the legacy DiskANN builder's node locks under nested
    // Rayon work stealing. Isolate fixture construction; search still uses the
    // normal worker pool after this scoped build finishes.
    let builder = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let result = builder.install(|| {
        build_diskann_index(
            flat,
            N_POINTS,
            DIM,
            1.15, // alpha
            GRAPH_DEGREE,
            BUILD_L,
            false, // build_pq
            None,  // n_subquantizers
            None,  // _n_bits
            true,  // compute_candidate_sets
            MAX_EXTRA,
        )
        .expect("build_diskann_index")
    });
    drop(builder);

    // Drop the diskann inner index — its slab is no longer needed once
    // we have the partitions.
    drop(result.index);

    let mut ds = InmemDataset::<f32, DIM>::new(N_POINTS, 1.0).unwrap();
    ds.data.memcpy(&flat[..N_POINTS * DIM]).unwrap();

    StaticOrion::<DIM>::new(
        ds,
        &result.partitions,
        result.entry_point,
        GRAPH_DEGREE,
        MAX_EXTRA,
    )
}

fn query_at(flat: &[f32], idx: usize) -> [f32; DIM] {
    let mut q = [0.0f32; DIM];
    q.copy_from_slice(&flat[idx * DIM..(idx + 1) * DIM]);
    q
}

// ── Tests ────────────────────────────────────────────────────────
//
// All scenarios run sequentially inside ONE `#[test]` function
// because the underlying `orion_ann_diskann::create_inmem_index` builder has
// global state (rayon pool init + shared static buffers) that races
// when multiple builds run in parallel under `cargo test`'s default
// multi-threaded harness. A single integration test sidesteps that
// without needing the `serial_test` crate.

#[test]
fn search_pipeline_e2e() {
    let flat = make_flat_dataset();
    let orion = build_orion_fixture(&flat);

    let admission = L2U8Admission::new(orion.ensure_quantized_dataset());
    let rerank = F32Rerank::new(orion.dataset());
    let no_pf: &Prefilter<'_> = &orion_ann_core::search::cascade::stage::NoPrefilter;

    // ── Scenario 1: self-query recovers self as top-1 ──
    for i in 0..8u32 {
        let q = query_at(&flat, i as usize);
        let neighbors = orion
            .search_observed(
                orion_ann_core::search::query::QueryInput::F32(&q),
                DynamicSearchConfig {
                    k: 5,
                    search_list_size: 32,
                    window_size: 5,
                    epsilon: 0.0,
                    early_exit_limit: usize::MAX,
                },
                DynamicCascade {
                    prefilter: no_pf,
                    admission: &admission,
                    rerank: &rerank,
                },
                orion_ann_core::search::diagnostics::NeighborMode::default(),
                &mut orion_ann_core::search::diagnostics::NoopObserver,
            )
            .expect("search");
        assert!(!neighbors.is_empty(), "vid={i}: empty result");
        assert_eq!(
            neighbors[0], i,
            "vid={i}: top-1 should be self, got {neighbors:?}"
        );
    }

    // ── Scenario 2: calibrate returns finite params ──
    let warmup: Vec<[f32; DIM]> = (0..16).map(|i| query_at(&flat, i)).collect();
    let calib = orion
        .calibrate(
            &warmup,
            /* sls */ 32,
            /* ws */ 5,
            Default::default(),
        )
        .expect("calibrate");
    assert!(calib.threshold.is_finite());
    assert!(calib.threshold >= 0.0 && calib.threshold <= 1.0);
    assert!(calib.early_exit_limit > 0);
    assert!(calib.early_exit_limit <= 32 * 4);

    // ── Scenario 3: batch search top-1 matches single search ──
    let queries: Vec<[f32; DIM]> = (0..4u32).map(|i| query_at(&flat, i as usize)).collect();
    let mut single_top1 = Vec::with_capacity(4);
    for q in queries.iter() {
        let r = orion
            .search_observed(
                orion_ann_core::search::query::QueryInput::F32(q),
                DynamicSearchConfig {
                    k: 3,
                    search_list_size: 32,
                    window_size: 5,
                    epsilon: 0.0,
                    early_exit_limit: usize::MAX,
                },
                DynamicCascade {
                    prefilter: no_pf,
                    admission: &admission,
                    rerank: &rerank,
                },
                orion_ann_core::search::diagnostics::NeighborMode::default(),
                &mut orion_ann_core::search::diagnostics::NoopObserver,
            )
            .unwrap();
        single_top1.push(r[0]);
    }
    let batch = orion
        .search_batch(
            orion_ann_core::search::query::QueryInput::F32(queries.as_flattened()),
            DynamicSearchConfig {
                k: 3,
                search_list_size: 32,
                window_size: 5,
                epsilon: 0.0,
                early_exit_limit: usize::MAX,
            },
            DynamicCascade {
                prefilter: no_pf,
                admission: &admission,
                rerank: &rerank,
            },
            None,
        )
        .expect("search_batch");
    assert_eq!(batch.len(), 4);
    for (i, row) in batch.rows().enumerate() {
        assert_eq!(
            row[0], single_top1[i],
            "query {i}: batch/single top-1 mismatch"
        );
    }

    // ── Scenario 4: non-zero epsilon path (DCC reconfigure) ──
    let q = query_at(&flat, 0);
    let r = orion
        .search_observed(
            orion_ann_core::search::query::QueryInput::F32(&q),
            DynamicSearchConfig {
                k: 5,
                search_list_size: 32,
                window_size: 5,
                epsilon: 0.15,
                early_exit_limit: 10,
            },
            DynamicCascade {
                prefilter: no_pf,
                admission: &admission,
                rerank: &rerank,
            },
            orion_ann_core::search::diagnostics::NeighborMode::default(),
            &mut orion_ann_core::search::diagnostics::NoopObserver,
        )
        .expect("search with convergence");
    assert!(!r.is_empty());
    assert_eq!(r[0], 0);

    // ── Scenario 5: distinct queries → distinct neighborhoods ──
    let r0 = orion
        .search_observed(
            orion_ann_core::search::query::QueryInput::F32(&query_at(&flat, 0)),
            DynamicSearchConfig {
                k: 10,
                search_list_size: 32,
                window_size: 5,
                epsilon: 0.0,
                early_exit_limit: usize::MAX,
            },
            DynamicCascade {
                prefilter: no_pf,
                admission: &admission,
                rerank: &rerank,
            },
            orion_ann_core::search::diagnostics::NeighborMode::default(),
            &mut orion_ann_core::search::diagnostics::NoopObserver,
        )
        .unwrap();
    let r5 = orion
        .search_observed(
            orion_ann_core::search::query::QueryInput::F32(&query_at(&flat, 5)),
            DynamicSearchConfig {
                k: 10,
                search_list_size: 32,
                window_size: 5,
                epsilon: 0.0,
                early_exit_limit: usize::MAX,
            },
            DynamicCascade {
                prefilter: no_pf,
                admission: &admission,
                rerank: &rerank,
            },
            orion_ann_core::search::diagnostics::NeighborMode::default(),
            &mut orion_ann_core::search::diagnostics::NoopObserver,
        )
        .unwrap();
    assert_ne!(
        r0, r5,
        "queries from different vertices returned identical top-10"
    );

    // ── Scenario 6: alternative L2-cascade flavours — exercises
    //    `ensure_quantized_dataset_l2_u16`, `_l2_kt`, `_jl`, the U16
    //    rerank, and the JL / RaBitQ Hamming prefilters. Each
    //    `ensure_*` call lazily builds + saves a sidecar so it
    //    measures both the build and the dispatch path.
    {
        let q = query_at(&flat, 1);
        let admission_u16 = L2U16Admission::new(orion.ensure_quantized_dataset_l2_u16());
        let rerank_u16 = U16Rerank::new(orion.ensure_quantized_dataset_l2_u16());
        let r = orion
            .search_observed(
                orion_ann_core::search::query::QueryInput::F32(&q),
                DynamicSearchConfig {
                    k: 3,
                    search_list_size: 32,
                    window_size: 5,
                    epsilon: 0.0,
                    early_exit_limit: usize::MAX,
                },
                DynamicCascade {
                    prefilter: no_pf,
                    admission: &admission_u16,
                    rerank: &rerank_u16,
                },
                orion_ann_core::search::diagnostics::NeighborMode::default(),
                &mut orion_ann_core::search::diagnostics::NoopObserver,
            )
            .expect("L2-U16 cascade");
        assert_eq!(r[0], 1);
    }

    {
        let q = query_at(&flat, 2);
        let kt_ds = orion.ensure_quantized_dataset_l2_kt();
        let admission_kt = L2KTAdmission::new(kt_ds);
        let r = orion
            .search_observed(
                orion_ann_core::search::query::QueryInput::F32(&q),
                DynamicSearchConfig {
                    k: 3,
                    search_list_size: 32,
                    window_size: 5,
                    epsilon: 0.0,
                    early_exit_limit: usize::MAX,
                },
                DynamicCascade {
                    prefilter: no_pf,
                    admission: &admission_kt,
                    rerank: &rerank,
                },
                orion_ann_core::search::diagnostics::NeighborMode::default(),
                &mut orion_ann_core::search::diagnostics::NoopObserver,
            )
            .expect("L2-KT cascade");
        assert_eq!(r[0], 2);
    }

    {
        // JL prefilter + L2-U8 admission + f32 rerank. The JL build
        // covers `jl_sparse_dataset::build_from`, the prefilter
        // session covers `prefilter/jl.rs::open`, and the streaming
        // distance covers the `JLHammingDistance` paths in
        // `orion-vector/distance_kernel.rs` + `distance_stream.rs`.
        let q = query_at(&flat, 3);
        let jl_ds = orion.ensure_quantized_dataset_jl();
        let pf = JlPrefilter::new(jl_ds);
        let filter: &Prefilter<'_> = &pf;
        let r = orion
            .search_observed(
                orion_ann_core::search::query::QueryInput::F32(&q),
                DynamicSearchConfig {
                    k: 3,
                    search_list_size: 32,
                    window_size: 5,
                    epsilon: 0.0,
                    early_exit_limit: usize::MAX,
                },
                DynamicCascade {
                    prefilter: filter,
                    admission: &admission,
                    rerank: &rerank,
                },
                orion_ann_core::search::diagnostics::NeighborMode::default(),
                &mut orion_ann_core::search::diagnostics::NoopObserver,
            )
            .expect("JL prefilter cascade");
        assert!(!r.is_empty());
    }

    // ── Scenario 7: MIPS cascade with IP rerank — exercises the
    //    inner-product path through `MipsI8Admission`, the
    //    `IpF32Rerank` impl, and `ensure_quantized_dataset_mips`.
    {
        let q = query_at(&flat, 4);
        let mips_ds = orion.ensure_quantized_dataset_mips();
        let admission_mips = MipsI8Admission::new(mips_ds);
        let rerank_ip = IpF32Rerank::new(orion.dataset());
        let r = orion
            .search_observed(
                orion_ann_core::search::query::QueryInput::F32(&q),
                DynamicSearchConfig {
                    k: 3,
                    search_list_size: 32,
                    window_size: 5,
                    epsilon: 0.0,
                    early_exit_limit: usize::MAX,
                },
                DynamicCascade {
                    prefilter: no_pf,
                    admission: &admission_mips,
                    rerank: &rerank_ip,
                },
                orion_ann_core::search::diagnostics::NeighborMode::default(),
                &mut orion_ann_core::search::diagnostics::NoopObserver,
            )
            .expect("MIPS cascade");
        assert!(!r.is_empty());
    }

    // ── Scenario 8: RaBitQ prefilter for the JL alternative ──
    {
        let q = query_at(&flat, 5);
        let rabitq_ds = orion.ensure_quantized_dataset_rabitq();
        let pf = RabitqPrefilter::new(rabitq_ds);
        let filter: &Prefilter<'_> = &pf;
        let r = orion
            .search_observed(
                orion_ann_core::search::query::QueryInput::F32(&q),
                DynamicSearchConfig {
                    k: 3,
                    search_list_size: 32,
                    window_size: 5,
                    epsilon: 0.0,
                    early_exit_limit: usize::MAX,
                },
                DynamicCascade {
                    prefilter: filter,
                    admission: &admission,
                    rerank: &rerank,
                },
                orion_ann_core::search::diagnostics::NeighborMode::default(),
                &mut orion_ann_core::search::diagnostics::NoopObserver,
            )
            .expect("RaBitQ prefilter cascade");
        assert!(!r.is_empty());
    }

    // ── Scenario 9: JL Hadamard prefilter ──
    {
        let q = query_at(&flat, 6);
        let jlh_ds = orion.ensure_quantized_dataset_jl_hadamard();
        let pf = JlHadamardPrefilter::new(jlh_ds);
        let filter: &Prefilter<'_> = &pf;
        let r = orion
            .search_observed(
                orion_ann_core::search::query::QueryInput::F32(&q),
                DynamicSearchConfig {
                    k: 3,
                    search_list_size: 32,
                    window_size: 5,
                    epsilon: 0.0,
                    early_exit_limit: usize::MAX,
                },
                DynamicCascade {
                    prefilter: filter,
                    admission: &admission,
                    rerank: &rerank,
                },
                orion_ann_core::search::diagnostics::NeighborMode::default(),
                &mut orion_ann_core::search::diagnostics::NoopObserver,
            )
            .expect("JL Hadamard prefilter cascade");
        assert!(!r.is_empty());
    }

    // ── Scenario 10: neighbor_contribution analysis ──
    //   `sort_neighbors_by_distance` walks every node × every neighbor
    //   and produces a distance-sorted adjacency. `greedy_search_truncated`
    //   runs a reference beam search on that adj list.
    //   `search_with_contribution` is the instrumented variant that
    //   tracks per-position admission stats. Together they cover the
    //   `analysis/neighbor_contribution.rs` file end-to-end.
    #[cfg(feature = "benchmark")]
    {
        use orion_ann_core::search::diagnostics::analysis::neighbor_contribution::{
            PositionContributionStats, greedy_search_truncated, search_with_contribution,
            sort_neighbors_by_distance,
        };

        let sorted_adj = sort_neighbors_by_distance(orion.graph(), &flat, DIM);
        assert_eq!(sorted_adj.len(), N_POINTS);
        // Every node should retain all its graph neighbors after sorting.
        for (node, nbrs) in sorted_adj.iter().enumerate() {
            assert_eq!(nbrs.len(), orion.graph().neighbors(node).len());
        }

        // Reference greedy search on the sorted adjacency.
        let entry = orion.entry_point();
        let q = query_at(&flat, 7);
        let r = greedy_search_truncated(
            &sorted_adj,
            &flat,
            DIM,
            entry,
            &q,
            5,
            32,
            /* max_nbrs */ usize::MAX,
        );
        assert!(!r.is_empty());

        // Same input, truncated to half the max neighbors — must still
        // return some result (likely lower recall, just check non-empty).
        let r_trunc = greedy_search_truncated(&sorted_adj, &flat, DIM, entry, &q, 5, 32, 4);
        assert!(!r_trunc.is_empty());

        // Instrumented variant that tracks per-position contribution.
        let mut stats = PositionContributionStats::new();
        let r_ctx = search_with_contribution(
            &sorted_adj,
            &flat,
            DIM,
            entry,
            &q,
            5,
            32,
            /* max_degree */ GRAPH_DEGREE as usize,
            &mut stats,
        );
        assert!(!r_ctx.is_empty());
        assert!(stats.num_queries >= 1);
        assert!(stats.total_expansions >= 1);
        // Display impl smoke (covers std::fmt::Display + the histogram).
        let formatted = format!("{stats}");
        assert!(formatted.contains("expansions"));
    }

    // ── Scenario 11: cliff_neighbor_stats analysis ──
    //   `compute_cliff_stats` walks every node and computes the
    //   distance-gap histogram across its neighbors. `annotate_bf_ranks`
    //   then attaches brute-force KNN ranks. `summarize_cliff_stats` +
    //   `summarize_cliff_ranks` produce aggregate summaries. Together
    //   they cover the 588-line `cliff_neighbor_stats.rs` file.
    #[cfg(feature = "benchmark")]
    {
        use orion_ann_core::search::diagnostics::analysis::cliff_neighbor_stats::{
            annotate_bf_ranks, compute_cliff_stats, summarize_cliff_ranks, summarize_cliff_stats,
        };

        let mut stats = compute_cliff_stats(orion.graph(), &flat, DIM);
        assert_eq!(stats.len(), N_POINTS);
        // Every node either has neighbors with valid cliff_pos or
        // degree==0 (which shouldn't happen on a Vamana graph but the
        // function handles it).
        for s in &stats {
            assert!(s.cliff_pos <= s.degree);
            assert!(s.min_distance <= s.max_distance);
        }

        // Annotate with brute-force ranks (capped at 32).
        annotate_bf_ranks(&mut stats, &flat, DIM, N_POINTS, /* bf_max_rank */ 32);

        // Aggregate summary.
        let summary = summarize_cliff_stats(&stats);
        assert_eq!(summary.num_nodes, N_POINTS);
        assert!(summary.avg_degree > 0.0);
        assert!(summary.mean_cliff_ratio.is_finite());
        // Display impl smoke.
        let formatted = format!("{summary}");
        assert!(formatted.contains("Cliff"));

        // Rank summary — `topk_thresholds` are the rank buckets used
        // for the % stats (top-1, top-5, top-10, top-20).
        let rank_summary = summarize_cliff_ranks(&stats, &[1, 5, 10, 20]);
        let rank_formatted = format!("{rank_summary}");
        assert!(!rank_formatted.is_empty());

        // Empty edge case — `summarize_cliff_stats` returns a zeroed
        // summary on an empty slice. Covers the early-return branch.
        let empty = summarize_cliff_stats(&[]);
        assert_eq!(empty.num_nodes, 0);
        assert_eq!(empty.avg_degree, 0.0);
    }

    // ── Scenario 12: remaining `ensure_quantized_dataset_*` dispatches.
    //   Exercises the MipsI16, RabitQ-B4, and JL-MIPS sidecar builders
    //   in `compressed_index.rs` that the other scenarios skipped.
    {
        use orion_ann_core::search::cascade::stage::admission::MipsI16Admission;

        let q = query_at(&flat, 8);

        // MipsI16 admission + IpF32 rerank.
        let mips_i16_ds = orion.ensure_quantized_dataset_mips_i16();
        let admission_mi16 = MipsI16Admission::new(mips_i16_ds);
        let rerank_ip = IpF32Rerank::new(orion.dataset());
        let r = orion
            .search_observed(
                orion_ann_core::search::query::QueryInput::F32(&q),
                DynamicSearchConfig {
                    k: 3,
                    search_list_size: 32,
                    window_size: 5,
                    epsilon: 0.0,
                    early_exit_limit: usize::MAX,
                },
                DynamicCascade {
                    prefilter: no_pf,
                    admission: &admission_mi16,
                    rerank: &rerank_ip,
                },
                orion_ann_core::search::diagnostics::NeighborMode::default(),
                &mut orion_ann_core::search::diagnostics::NoopObserver,
            )
            .expect("MipsI16 cascade");
        assert!(!r.is_empty());

        // Touch the B=4 RaBitQ and JL-MIPS sidecars too — the
        // `ensure_*` returns build the sidecar on first call.
        let _b4 = orion.ensure_quantized_dataset_rabitq_b4();
        let _jl_mips = orion.ensure_quantized_dataset_jl_mips();
    }

    // ── Scenario 13: calibrate_with_diagnostics ──
    //   Calibrate has two entry points — `calibrate` and the more
    //   verbose `calibrate_with_diagnostics`. The latter populates a
    //   diagnostics struct with per-step histograms.
    #[cfg(feature = "benchmark")]
    {
        let warmup: Vec<[f32; DIM]> = (0..16).map(|i| query_at(&flat, i)).collect();
        let diag = orion
            .calibrate_with_diagnostics(
                &warmup,
                /* sls */ 32,
                /* ws */ 5,
                Default::default(),
            )
            .expect("calibrate_with_diagnostics");
        assert!(diag.params.threshold.is_finite());
        assert!(diag.params.early_exit_limit > 0);
    }

    // ── Scenario 14: diagnostic / profile search variants ──
    //   `search_diag` returns extra step metadata; `search_profile`
    //   returns deeper per-stage profile stats. Both live in
    //   `search/utils.rs` and aren't on the cascade path
    //   the unified search exercises.
    #[cfg(feature = "benchmark")]
    {
        let q = query_at(&flat, 9);
        let (results, _converge_step, total_steps, phase1_ndc, phase2_ndc) = orion
            .search_diag(&q, 5, 32, 5, 0.0, usize::MAX)
            .expect("search_diag");
        assert!(!results.is_empty());
        assert!(total_steps > 0);
        assert!(phase1_ndc + phase2_ndc > 0);

        let queries = [query_at(&flat, 9), query_at(&flat, 10)];
        let stats = orion
            .search_profile(&queries, 5, 32, 5, 0.0)
            .expect("search_profile");
        assert!(stats.total_ns() > 0);
    }

    // ── Scenario 15: ADSampling admission ──
    //   `AdsF32Admission` wraps the raw f32 dataset with a probabilistic
    //   early-abandon `distance_compare_adsampling` predicate. Covers
    //   `stage/admission/ads_f32.rs` (the only 0%-coverage admission
    //   left) plus the ADSampling path inside
    //   `FullPrecisionDistance::distance_compare_adsampling` for f32.
    #[cfg(feature = "benchmark")]
    {
        use orion_ann_core::search::cascade::stage::admission::ads_f32::AdsF32Admission;

        let q = query_at(&flat, 10);
        let admission_ads = AdsF32Admission::new(orion.dataset(), /* ads_epsilon */ 2.1);
        let r = orion
            .search_observed(
                orion_ann_core::search::query::QueryInput::F32(&q),
                DynamicSearchConfig {
                    k: 3,
                    search_list_size: 32,
                    window_size: 5,
                    epsilon: 0.0,
                    early_exit_limit: usize::MAX,
                },
                DynamicCascade {
                    prefilter: no_pf,
                    admission: &admission_ads,
                    rerank: &rerank,
                },
                orion_ann_core::search::diagnostics::NeighborMode::default(),
                &mut orion_ann_core::search::diagnostics::NoopObserver,
            )
            .expect("ADS admission cascade");
        assert!(!r.is_empty());
    }

    // ── Scenario 16: JL-MIPS prefilter ──
    //   Covers the `JlMipsPrefilter` half of `prefilter/jl.rs` (the
    //   non-mips half was hit in Scenario 6). Uses the JL-MIPS sidecar
    //   built lazily in Scenario 12.
    {
        use orion_ann_core::search::cascade::stage::prefilter::JlMipsPrefilter;
        let q = query_at(&flat, 11);
        let jl_mips_ds = orion.ensure_quantized_dataset_jl_mips();
        let pf = JlMipsPrefilter::new(jl_mips_ds);
        let filter: &Prefilter<'_> = &pf;
        let admission_mips = MipsI8Admission::new(orion.ensure_quantized_dataset_mips());
        let rerank_ip = IpF32Rerank::new(orion.dataset());
        let r = orion
            .search_observed(
                orion_ann_core::search::query::QueryInput::F32(&q),
                DynamicSearchConfig {
                    k: 3,
                    search_list_size: 32,
                    window_size: 5,
                    epsilon: 0.0,
                    early_exit_limit: usize::MAX,
                },
                DynamicCascade {
                    prefilter: filter,
                    admission: &admission_mips,
                    rerank: &rerank_ip,
                },
                orion_ann_core::search::diagnostics::NeighborMode::default(),
                &mut orion_ann_core::search::diagnostics::NoopObserver,
            )
            .expect("JL-MIPS prefilter cascade");
        assert!(!r.is_empty());
    }

    // Scenario 17: the graph codec preserves search results across a round trip.
    {
        let mut meta = Vec::new();
        let mut writer = std::io::BufWriter::new(Vec::new());
        orion.encode_native_graph(&mut meta, &mut writer).unwrap();
        let graph = writer.into_inner().unwrap();

        // Reload with the same f32 dataset attached.
        let mut empty_ds = InmemDataset::<f32, DIM>::new(N_POINTS, 1.0).unwrap();
        empty_ds.data.memcpy(&flat[..N_POINTS * DIM]).unwrap();
        let reloaded = StaticOrion::<DIM>::decode_native_graph(
            empty_ds,
            meta.as_slice(),
            graph.as_slice(),
            graph.len() as u64,
        )
        .expect("decode graph");

        // Search through both — top-1 must agree on the same query.
        let q = query_at(&flat, 12);
        let r_orig = orion
            .search_observed(
                orion_ann_core::search::query::QueryInput::F32(&q),
                DynamicSearchConfig {
                    k: 5,
                    search_list_size: 32,
                    window_size: 5,
                    epsilon: 0.0,
                    early_exit_limit: usize::MAX,
                },
                DynamicCascade {
                    prefilter: no_pf,
                    admission: &admission,
                    rerank: &rerank,
                },
                orion_ann_core::search::diagnostics::NeighborMode::default(),
                &mut orion_ann_core::search::diagnostics::NoopObserver,
            )
            .unwrap();
        // Rebuild admission + rerank against the reloaded instance.
        let admission2 = L2U8Admission::new(reloaded.ensure_quantized_dataset());
        let rerank2 = F32Rerank::new(reloaded.dataset());
        let r_reloaded = reloaded
            .search_observed(
                orion_ann_core::search::query::QueryInput::F32(&q),
                DynamicSearchConfig {
                    k: 5,
                    search_list_size: 32,
                    window_size: 5,
                    epsilon: 0.0,
                    early_exit_limit: usize::MAX,
                },
                DynamicCascade {
                    prefilter: no_pf,
                    admission: &admission2,
                    rerank: &rerank2,
                },
                orion_ann_core::search::diagnostics::NeighborMode::default(),
                &mut orion_ann_core::search::diagnostics::NoopObserver,
            )
            .unwrap();
        assert_eq!(r_orig[0], r_reloaded[0], "top-1 mismatch after reload");
    }

    // ── Scenario 18: dataset codec save/load round trips ──
    //   Each of the JL-Hadamard, L2-KT, and JL-MIPS sidecars has its
    //   own on-disk format. The `ensure_quantized_dataset_*` paths
    //   above already built them in memory; here we cover the
    //   explicit save() + load() round-trip code that the production
    //   builder uses on cache miss / cache hit.
    {
        use orion_ann_core::model::dataset::JLSparseDataset;
        use orion_ann_core::model::dataset::jl_hadamard_dataset::JlHadamardDataset;
        use orion_ann_core::model::dataset::l2_kt_dataset::L2KTDataset;

        // JL Hadamard round trip.
        let tmp_jlh = std::env::temp_dir().join("orion_e2e_jl_hadamard.qjlh");
        let _ = std::fs::remove_file(&tmp_jlh);
        let jlh = JlHadamardDataset::build_with_padding(orion.dataset(), 0x42, 1024);
        jlh.encode(&mut std::fs::File::create(&tmp_jlh).unwrap())
            .expect("save jl_hadamard");
        let _loaded =
            sidecar_file::decode(&tmp_jlh, JlHadamardDataset::decode).expect("load jl_hadamard");
        let _ = std::fs::remove_file(&tmp_jlh);

        // L2-KT round trip.
        let tmp_kt = std::env::temp_dir().join("orion_e2e_l2_kt.qdsl2kt");
        let _ = std::fs::remove_file(&tmp_kt);
        let kt = L2KTDataset::build_from(orion.dataset());
        kt.encode(&mut std::fs::File::create(&tmp_kt).unwrap())
            .expect("save l2_kt");
        let _loaded = sidecar_file::decode(&tmp_kt, L2KTDataset::decode).expect("load l2_kt");
        let _ = std::fs::remove_file(&tmp_kt);

        // JL Sparse round trip.
        let tmp_jl = std::env::temp_dir().join("orion_e2e_jl_sparse.jls");
        let _ = std::fs::remove_file(&tmp_jl);
        let jl = JLSparseDataset::<1024>::build_from(orion.dataset(), 0x42);
        jl.encode(&mut std::fs::File::create(&tmp_jl).unwrap())
            .expect("save jl_sparse");
        let _loaded =
            sidecar_file::decode(&tmp_jl, JLSparseDataset::<1024>::decode).expect("load jl_sparse");
        let _ = std::fs::remove_file(&tmp_jl);
    }

    // ── Scenario 19: search loop variations ──
    //   Drive the unified beam loop at several `search_list_size`
    //   values (small / large / equal to N_POINTS) so the inner
    //   admission + convergence + early-exit branches each get
    //   exercised on a wider variety of pq sizes.
    {
        let q = query_at(&flat, 13);
        for &sls in &[8usize, 16, 48, 96] {
            let r = orion
                .search_observed(
                    orion_ann_core::search::query::QueryInput::F32(&q),
                    DynamicSearchConfig {
                        k: 3,
                        search_list_size: sls,
                        window_size: 5,
                        epsilon: 0.0,
                        early_exit_limit: usize::MAX,
                    },
                    DynamicCascade {
                        prefilter: no_pf,
                        admission: &admission,
                        rerank: &rerank,
                    },
                    orion_ann_core::search::diagnostics::NeighborMode::default(),
                    &mut orion_ann_core::search::diagnostics::NoopObserver,
                )
                .expect("search loop variation");
            assert!(!r.is_empty(), "sls={sls}: empty result");
        }
        // Also exercise the early-exit path with a tight limit and the
        // convergence path with a non-trivial epsilon together — covers
        // the branch in the inner loop where both signals fire.
        let r = orion
            .search_observed(
                orion_ann_core::search::query::QueryInput::F32(&q),
                DynamicSearchConfig {
                    k: 3,
                    search_list_size: 32,
                    window_size: 5,
                    epsilon: 0.20,
                    early_exit_limit: 5,
                },
                DynamicCascade {
                    prefilter: no_pf,
                    admission: &admission,
                    rerank: &rerank,
                },
                orion_ann_core::search::diagnostics::NeighborMode::default(),
                &mut orion_ann_core::search::diagnostics::NoopObserver,
            )
            .expect("search with both convergence and early exit");
        assert!(!r.is_empty());
    }

    // ── Scenario 20: SearchProfileStats helpers ──
    //   `total_ns` and `print_report` aren't on the cascade hot path;
    //   the calibrate / profile entry points don't always touch them.
    //   Touch them directly to cover the formatting + helper code.
    #[cfg(feature = "benchmark")]
    {
        let queries = [query_at(&flat, 14), query_at(&flat, 15)];
        let stats = orion
            .search_profile(&queries, 5, 32, 5, 0.0)
            .expect("search_profile");
        let total = stats.total_ns();
        assert!(total > 0);
        // `print_report` writes to stdout; we just want to exercise
        // the code path. The println output is captured by the test
        // harness and discarded on pass.
        stats.print_report();
    }
}
