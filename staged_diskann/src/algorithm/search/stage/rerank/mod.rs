/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Rerank stage
//!
//! Final precision pass at end-of-beam. Takes the top
//! `k · rerank_factor` PQ entries, recomputes their distance at higher
//! precision (f32 truth or u16 truth), sorts by the new distance, and
//! emits the top-k vertex IDs.
//!
//! Implementations:
//!
//! * [`f32_truth::F32Rerank`] — full f32 base. Wraps the existing
//!   `super::super::rerank::run_f32_truth`. Default for L2-Q.
//! * [`u16_truth::U16Rerank`] — u16 sidecar. Wraps
//!   `super::super::rerank::run_u16_truth`. PA's `quantize_bits=16`
//!   shape.
//! * [`NoRerank`] (below) — top-k directly from PQ. Useful when the
//!   admission tier ranks at sufficient precision (e.g. f32
//!   admission or RaBitQ B=4 with per-vertex correction).
//!
//! The rerank stage is called **once per query** on a small candidate
//! set (typically `k · 2 = 20` vertices), so its dispatch cost is
//! negligible.

pub mod f32_truth;
pub mod ip_f32_truth;
pub mod u16_truth;

pub use f32_truth::F32Rerank;
pub use ip_f32_truth::IpF32Rerank;
pub use u16_truth::U16Rerank;

use crate::model::Neighbor;
use crate::model::scratch::InMemSearchScratch;

/// Final precision pass on the top-(k · rerank_factor) PQ entries.
pub trait RerankStage<const N: usize>: Send + Sync {
    /// Score and emit top-k.
    ///
    /// * `query` — the original f32 query (pre-quantization).
    /// * `pq_entries` — top entries from the PQ at end of beam.
    ///   Already in distance-ascending order. Pass the first
    ///   `min(k · rerank_factor, pq_entries.len())` to the rerank
    ///   logic.
    /// * `scratch` — reused buffers for the rerank's id-list staging
    ///   and padded-query allocation.
    fn rerank(
        &self,
        query: &[f32; N],
        pq_entries: &[Neighbor],
        k: usize,
        rerank_factor: usize,
        scratch: &mut InMemSearchScratch,
    ) -> Vec<u32>;
}

/// Top-k directly from the PQ without any rerank pass.
///
/// Use when the admission tier already ranks at sufficient precision
/// (e.g. f32 admission, or when calibrated quality margins make
/// rerank a no-op).
pub struct NoRerank;

impl<const N: usize> RerankStage<N> for NoRerank {
    fn rerank(
        &self,
        _query: &[f32; N],
        pq_entries: &[Neighbor],
        k: usize,
        _rerank_factor: usize,
        _scratch: &mut InMemSearchScratch,
    ) -> Vec<u32> {
        let n_out = k.min(pq_entries.len());
        let mut out = Vec::with_capacity(n_out);
        for e in pq_entries.iter().take(n_out) {
            out.push(e.id);
        }
        out
    }
}
