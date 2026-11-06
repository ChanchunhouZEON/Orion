/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use crate::StaticOrion;
use crate::{Error, Result};

use orion_ann_vector::FullPrecisionDistance;

// Retain existing type paths for downstream callers.
#[cfg(feature = "benchmark")]
pub use super::diagnostics::{SearchProfile, SearchProfileStats};

// ── Search-loop tuning constants ──────────────────────────────────────────
// Admission/prefilter and rerank use the same 32-cache-line near window.
// These defaults match the validated near32/burst32 stream schedule. Units are
// cache lines, not vector counts; no CPU-cache residency is implied.
const DEFAULT_NEAR_LINES: usize = 32;

/// Near lookahead for quantized stages. Benchmark builds may override it;
/// ordinary Rust/Python/CLI consumers always use the measured default.
#[cfg(feature = "benchmark")]
pub(super) fn dstream_la_q() -> usize {
    use std::sync::OnceLock;
    static CACHE: OnceLock<usize> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("ORION_DSTREAM_LA_Q")
            .ok()
            .or_else(|| std::env::var("ORION_DSTREAM_LOOKAHEAD").ok())
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v <= 128)
            .unwrap_or(DEFAULT_NEAR_LINES)
    })
}

#[cfg(not(feature = "benchmark"))]
pub(super) fn dstream_la_q() -> usize {
    DEFAULT_NEAR_LINES
}

/// Near lookahead for reranking; measured in cache lines just like admission.
#[cfg(feature = "benchmark")]
pub(super) fn dstream_la_truth() -> usize {
    use std::sync::OnceLock;
    static CACHE: OnceLock<usize> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("ORION_DSTREAM_LA_TRUTH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v <= 256)
            .unwrap_or(DEFAULT_NEAR_LINES)
    })
}

#[cfg(not(feature = "benchmark"))]
pub(super) fn dstream_la_truth() -> usize {
    DEFAULT_NEAR_LINES
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

use crate::model::dataset::quantized_dataset::pq::FixedChunkPQTable;
use std::sync::Arc;

impl<const N: usize> StaticOrion<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Compute PQ distance for a point given its ID and pre-computed chunk distances.
    /// SAFETY: Should be invoked only when pq is activated.
    #[inline]
    #[allow(dead_code)]
    pub(super) fn pq_distance(&self, point_id: u32, pq_dists: &[f32]) -> Result<f32> {
        let (pq, pq_codes, num_pq_chunks) = self.get_unwrapped_pq_component()?;

        let idx = point_id as usize;
        let code_start = idx * num_pq_chunks;
        let code = &pq_codes[code_start..code_start + num_pq_chunks];
        Ok(pq.adc_distance(code, pq_dists))
    }

    pub(super) fn get_unwrapped_pq_component(
        &self,
    ) -> Result<(&Arc<FixedChunkPQTable>, &Vec<u8>, usize)> {
        let pq = self.pq.as_ref().ok_or_else(|| {
            Error::invalid_state("Fixed Chunk PQ Table is None for now".to_string())
        })?;
        let pq_codes = self
            .pq_codes
            .as_ref()
            .ok_or_else(|| Error::invalid_state("PQ codes is None for now".to_string()))?;
        let num_pq_chunks = self.num_pq_chunks.ok_or_else(|| {
            Error::invalid_state("Number of pq chunks is None for now".to_string())
        })?;

        Ok((pq, pq_codes, num_pq_chunks))
    }
}

// Keep diagnostics bounded: do not format vector payloads or working buffers.
impl<const N: usize> std::fmt::Debug for AlignedQuery<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedQuery")
            .field("dimension", &(N))
            .finish_non_exhaustive()
    }
}
