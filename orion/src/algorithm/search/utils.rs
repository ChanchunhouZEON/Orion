/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use crate::Orion;
use diskann::common::ANNResult;
use vector::FullPrecisionDistance;

// Retain existing type paths for downstream callers.
pub use super::diagnostics::{SearchProfile, SearchProfileStats};

// ── Search-loop tuning constants ──────────────────────────────────────────
/// Cache-line lookahead for the software-pipelined `DistanceStream`
/// **Stage-1 quantized prefilter**. Stage-1 vert sizes:
/// glove100 i8 = **1 line/vert** (LA=12 ⇒ 12 verts ahead),
/// glove100 i16 = 2 lines/vert (LA=12 ⇒ 6 verts ahead),
/// SIFT u8 = 1 line/vert. Steady-state issues `LA_Q` prfm per outer
/// iter; M2 MSHR ≈ 12. Override via `ORION_DSTREAM_LA_Q=<N>`
/// (back-compat: `ORION_DSTREAM_LOOKAHEAD` still honored).
pub(super) fn dstream_la_q() -> usize {
    use std::sync::OnceLock;
    static CACHE: OnceLock<usize> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("ORION_DSTREAM_LA_Q")
            .ok()
            .or_else(|| std::env::var("ORION_DSTREAM_LOOKAHEAD").ok())
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v <= 128)
            // 64 chosen via LA × L sweep on glove100 i8 with the
            // sliding-window continuous-stride prefetch design.
            // LA=64 wins low/mid band (L=16=204k, L=32=126k, L=56=86k)
            // and ties LA=48/128 at high L. Yields 1.34× geomean PA
            // across 7 recall-aligned anchor points. Larger LA
            // (96/128) over-prefetches at narrow beam (L=16); smaller
            // LA (8/16) under-fills the MSHR queue.
            .unwrap_or(10)
    })
}

/// Cache-line lookahead for the software-pipelined `DistanceStream`
/// **Stage-2 f32 truth**. f32 vert sizes are larger:
/// glove100 f32 = 4 lines/vert (LA=12 ⇒ **3 verts ahead**),
/// SIFT f32 = 4 lines/vert, GIST f32 = **30 lines/vert** (LA=12 ⇒
/// 0 verts — LA is sub-vert here, prologue still primes lines for
/// the in-progress vert). Stage-2's per-vert compute (~25 ns @ N=100,
/// 4-acc kernel) is ~4× longer than Stage-1's, so the same line
/// budget translates to similar runway in nanoseconds.
/// Override via `ORION_DSTREAM_LA_TRUTH=<N>`.
pub(super) fn dstream_la_truth() -> usize {
    use std::sync::OnceLock;
    static CACHE: OnceLock<usize> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("ORION_DSTREAM_LA_TRUTH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v <= 256)
            // 64 — same default as `dstream_la_q`. Sliding-window
            // continuous-stride prefetch in `DistanceStream::run`
            // works the same way for Stage-1 quantized and Stage-2
            // f32 truth: one prfm per outer iter at distance
            // `lookahead_lines` ahead, keeping MSHR at steady state
            // without burst eviction. Larger LA gives a deeper
            // runway to hide DRAM latency on the f32 truth read,
            // which has the same per-line miss cost as Stage-1.
            .unwrap_or(6)
    })
}

/// Flush cadence by phase: `FLUSH_INTERVAL[converged as usize]` hops.
/// Pre-converged flushes every hop to keep pq_worst tight; converged
/// accumulates across 4 hops to amortize sort+merge over sparse admits
/// without letting pq_worst drift enough to defeat `early_exit`.
pub(super) const FLUSH_INTERVAL: [usize; 2] = [1, 4];

/// 3-way merge routing divisors re-fit from `pq_merge_bench` under
/// the pad16 PQ layout:
///   K * 8   < L → per-insert   (K < L / 8,    small-batch regime)
///   K * 1.5 > L → linear merge (K > L · 0.67, near-or-over-capacity)
///   otherwise   → gallop merge (log + bulk-memcpy wins in the middle)
///
/// The bench shows gallop owns the entire `K/L ∈ [0.125, 0.67]` band
/// for L ≥ 64 — its "binary-search insertion + extend_from_slice run
/// of cache-line-sized memcpys" pattern beats both per-insert
/// (O(K·L)) and linear merge (3-way branchy set-union) by 10-30 ns
/// per call. Per-insert wins below the 1/8 line because its constant
/// factor is just one binary-search + one copy_within with no scratch
/// swap. Linear merge wins above the 2/3 line because gallop's
/// `partition_point` degenerates to length-1 runs when admits are
/// near-uniform across the full PQ range.
///
/// Encoded as pure shifts + adds — both comparisons lower to one or
/// two shifts + an add + a compare, no integer multiply:
///   K * 8   = K << 3
///   K * 1.5 = K + (K >> 1)
const INSERT_ROUTE_SHIFT: u32 = 3; // << 3 = × 8

#[inline(always)]
pub(super) const fn insert_route_mul(k: usize) -> usize {
    k << INSERT_ROUTE_SHIFT
}

#[inline(always)]
pub(super) const fn linear_merge_mul(k: usize) -> usize {
    k + (k >> 1)
}

/// 16-byte aligned query buffer for efficient NEON loads.
/// Query is copied once at search entry, then reused for all distance computations.
#[repr(C, align(16))]
pub struct AlignedQuery<const N: usize>(pub [f32; N]);

// ── PQ helpers (formerly in utils.rs) ──────────────────────
use diskann::common::ANNError;
use diskann::model::FixedChunkPQTable;
use std::sync::Arc;

impl<const N: usize> Orion<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Compute PQ distance for a point given its ID and pre-computed chunk distances.
    /// SAFETY: Should be invoked only when pq is activated.
    #[inline]
    #[allow(dead_code)]
    pub(super) fn pq_distance(&self, point_id: u32, pq_dists: &[f32]) -> ANNResult<f32> {
        let (pq, pq_codes, num_pq_chunks) = self.get_unwrapped_pq_component()?;

        let idx = point_id as usize;
        let code_start = idx * num_pq_chunks;
        let code = &pq_codes[code_start..code_start + num_pq_chunks];
        Ok(pq.adc_distance(code, pq_dists))
    }

    pub(super) fn get_unwrapped_pq_component(
        &self,
    ) -> ANNResult<(&Arc<FixedChunkPQTable>, &Vec<u8>, usize)> {
        let pq = self.pq.as_ref().ok_or_else(|| {
            ANNError::log_pq_error("Fixed Chunk PQ Table is None for now".to_string())
        })?;
        let pq_codes = self
            .pq_codes
            .as_ref()
            .ok_or_else(|| ANNError::log_pq_error("PQ codes is None for now".to_string()))?;
        let num_pq_chunks = self.num_pq_chunks.ok_or_else(|| {
            ANNError::log_pq_error("Number of pq chunks is None for now".to_string())
        })?;

        Ok((pq, pq_codes, num_pq_chunks))
    }
}
