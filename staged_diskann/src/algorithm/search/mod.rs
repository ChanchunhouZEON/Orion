/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Search-module common infrastructure
//!
//! Shared per-thread metric machinery — counter handles like
//! [`VISIT_COUNT`] / [`NDC_I8`] / [`SETUP_NS`] live here so that every
//! search variant ([`in_mem_search::search_unified`], the stage trait
//! implementations under [`stage`], the diagnostics in [`utils`]) can
//! bump the same per-worker u64 slots without coupling.
//!
//! Search-loop tuning constants (`FLUSH_INTERVAL`, env-driven
//! lookahead, etc.) and diagnostic helpers stay in [`utils`].

use std::cell::UnsafeCell;

// ── Per-phase instrumentation counters (shared across search paths) ──
//
// **All 9 metrics packed into one cache line per worker thread.**
// Each rayon worker writes into `METRICS.slots[rayon_idx]`, a single
// 128-B aligned struct containing every per-query stat in 72 bytes of
// `u64` payload + 56 bytes of trailing pad. Two wins simultaneously:
//
//   1. **No false sharing** — distinct worker slots live on distinct
//      cache lines, so no coherence ping-pong between workers.
//   2. **Warm cache per worker** — one cache line per worker carries
//      every metric, so the line a worker touches to bump `visits` is
//      the same one it touches to bump `ndc_i8`, `query_count`, etc.
//      One L1 line hit per query instead of 7-9.
//
// Aggregation (`.sum()`, `.drain()`) walks all worker slots and sums
// them; only the benchmark's reporting code calls this, and only
// after the par_iter has joined (so there are no concurrent writers
// at read time).
//
// Sized for up to `MAX_WORKERS = 64` rayon threads — covers M2 Pro
// (8 cores), M2 Ultra (24), and well beyond. At 128 B per worker ×
// 64 workers = 8 KiB total static.
//
// Public API is unchanged: each metric is a zero-sized handle type
// (`VisitCount`, `NdcI8`, ...) with `.add(n)` / `.sum()` / `.drain()`
// / `.reset()` exactly as the previous sharded counter exposed, so
// every call site keeps writing `VISIT_COUNT.add(n)` unmodified.

/// Maximum supported rayon worker count. Calls from a thread whose
/// `rayon::current_thread_index()` exceeds this fall back to slot 0
/// (with a `debug_assert` — production runs always stay well under).
pub const MAX_WORKERS: usize = 64;

/// One cache-line-aligned struct holding **every** per-query metric
/// for a single worker thread. The 9 `u64` fields pack into the first
/// 72 bytes of the 128-byte cache line; the trailing 56 bytes are
/// padding kept by `#[repr(align(128))]` to keep adjacent worker
/// slots on distinct cache lines (anti-false-sharing).
#[repr(align(128))]
pub struct PerThreadMetrics {
    pub visit_count: UnsafeCell<u64>,
    /// Total graph neighbours expanded **before** the prefilter compaction
    /// (or = `visit_count` when no prefilter is active). Subtract
    /// `visit_count` from this to get the number filtered out, then
    /// divide by `raw_visit_count` to get the filter ratio.
    pub raw_visit_count: UnsafeCell<u64>,
    pub query_count: UnsafeCell<u64>,
    pub pre_conv_hops: UnsafeCell<u64>,
    pub post_conv_hops: UnsafeCell<u64>,
    pub pre_conv_admits: UnsafeCell<u64>,
    pub post_conv_admits: UnsafeCell<u64>,
    pub ndc_i8: UnsafeCell<u64>,
    pub ndc_f32: UnsafeCell<u64>,
    pub setup_ns: UnsafeCell<u64>,
}

impl PerThreadMetrics {
    pub const fn new() -> Self {
        Self {
            visit_count: UnsafeCell::new(0),
            raw_visit_count: UnsafeCell::new(0),
            query_count: UnsafeCell::new(0),
            pre_conv_hops: UnsafeCell::new(0),
            post_conv_hops: UnsafeCell::new(0),
            pre_conv_admits: UnsafeCell::new(0),
            post_conv_admits: UnsafeCell::new(0),
            ndc_i8: UnsafeCell::new(0),
            ndc_f32: UnsafeCell::new(0),
            setup_ns: UnsafeCell::new(0),
        }
    }
}

/// SAFETY: each worker slot is only ever written by the thread whose
/// `rayon::current_thread_index()` equals its array index. Reads
/// happen only from the aggregating thread (the benchmark driver),
/// and only after `par_iter` has joined — i.e. after a happens-before
/// edge from every writer.
unsafe impl Sync for PerThreadMetrics {}

pub struct MetricsTable {
    pub slots: [PerThreadMetrics; MAX_WORKERS],
}

impl MetricsTable {
    pub const fn new() -> Self {
        Self {
            slots: [const { PerThreadMetrics::new() }; MAX_WORKERS],
        }
    }
}

/// Global metrics table. Always reference fields via the `*_COUNT` /
/// `NDC_*` / `SETUP_NS` handles below; raw access is exposed only so
/// the benchmark's per-thread breakdown helper can walk slots without
/// a per-field accessor.
pub static METRICS: MetricsTable = MetricsTable::new();

/// Generates a zero-sized accessor handle bound to one field of
/// [`PerThreadMetrics`]. Preserves the previous public API
/// (`.add`/`.sum`/`.drain`/`.reset`) so call sites are unchanged.
macro_rules! define_metric {
    ($static_name:ident, $type_name:ident, $field:ident) => {
        pub struct $type_name;

        impl $type_name {
            /// Bump the calling worker's slot by `n`. Outside a rayon
            /// pool, falls back to slot 0 (debug-asserted single-writer).
            #[inline]
            pub fn add(&self, n: u64) {
                let idx = rayon::current_thread_index().unwrap_or(0);
                debug_assert!(
                    idx < MAX_WORKERS,
                    "rayon worker count exceeded MAX_WORKERS={MAX_WORKERS}"
                );
                unsafe {
                    *METRICS.slots[idx].$field.get() += n;
                }
            }

            /// Sum all per-worker slots. Safe only after all writers
            /// have joined (outside the `par_iter` scope).
            pub fn sum(&self) -> u64 {
                METRICS
                    .slots
                    .iter()
                    .map(|s| unsafe { *s.$field.get() })
                    .sum()
            }

            /// Sum then zero — mirrors the old `swap(0, Relaxed)`
            /// pattern used by `staged_sweep` between trials.
            pub fn drain(&self) -> u64 {
                let s = self.sum();
                self.reset();
                s
            }

            /// Zero every per-worker slot for this metric.
            pub fn reset(&self) {
                for slot in METRICS.slots.iter() {
                    unsafe {
                        *slot.$field.get() = 0;
                    }
                }
            }
        }

        pub static $static_name: $type_name = $type_name;
    };
}

define_metric!(VISIT_COUNT, VisitCount, visit_count);
// RAW_VISIT_COUNT — sum of `id_scratch.len()` captured **before** the
// prefilter compaction (= VISIT_COUNT when no prefilter is active).
// `filter_ratio = 1 - VISIT_COUNT / RAW_VISIT_COUNT`.
define_metric!(RAW_VISIT_COUNT, RawVisitCount, raw_visit_count);
define_metric!(QUERY_COUNT, QueryCount, query_count);
define_metric!(PRE_CONV_HOPS, PreConvHops, pre_conv_hops);
define_metric!(POST_CONV_HOPS, PostConvHops, post_conv_hops);
define_metric!(PRE_CONV_ADMITS, PreConvAdmits, pre_conv_admits);
define_metric!(POST_CONV_ADMITS, PostConvAdmits, post_conv_admits);
// NDC_I8 / NDC_F32 — total distance computations per query: every
// Stage-1 quantized distance (one per unseen neighbour per hop) +
// every Stage-2 f32 truth/rerank distance. Mirrors PA's `average cmps`.
define_metric!(NDC_I8, NdcI8, ndc_i8);
define_metric!(NDC_F32, NdcF32, ndc_f32);
// SETUP_NS — total nanoseconds spent in per-query setup (normalise +
// quantize query + scratch acquire/prepare/reconfigure + entry insert),
// summed across all threads. Divide by QUERY_COUNT.sum() for per-query
// setup cost.
define_metric!(SETUP_NS, SetupNs, setup_ns);

// ── Ablation toggle: include the per-node `extras` zone during the
// post-convergence rerank pass? Defaults to `true` (production
// behaviour: rerank walks `local + extra`). The ablation harness flips
// this to `false` for the "no-extras" variant so the SAME built graph
// (with its full extras zone stored) can serve both the full and the
// no-extras variants — eliminating the build-time topology confound
// of the earlier `max_extra = 0` rebuild approach. Process-global,
// not per-query: ablation variants run sequentially.
//
// Only the search call sites in `in_mem_search.rs` and `utils.rs`
// consult this — graph build, partition extraction, sidecar
// materialisation are unaffected. Set via [`set_include_extras`].
use std::sync::atomic::{AtomicBool, Ordering};
static INCLUDE_EXTRAS: AtomicBool = AtomicBool::new(true);

/// Override whether post-convergence rerank visits the extras zone.
/// Pass `false` for the "no-extras" ablation variant; `true` (default)
/// otherwise. Production code never calls this — leave the default.
#[inline]
pub fn set_include_extras(v: bool) {
    INCLUDE_EXTRAS.store(v, Ordering::Relaxed);
}

/// Read the current "include extras during rerank" flag.
#[inline]
pub fn include_extras() -> bool {
    INCLUDE_EXTRAS.load(Ordering::Relaxed)
}

pub mod in_mem_search;
pub mod utils;

pub mod async_beam_search;

pub mod convergence;

pub mod early_exit;

pub mod calibrate;

pub mod jl_hamming_cache;

pub mod stage;

pub use utils::SearchProfile;
