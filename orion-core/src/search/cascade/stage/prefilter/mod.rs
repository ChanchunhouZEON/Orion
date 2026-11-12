/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Prefilter stage
//!
//! Optional cheap rejection tier between graph-neighbour expansion
//! and PQ admission. Maintains a per-query running-mean threshold and
//! compacts the unseen-id list in place to only the candidates that
//! pass.
//!
//! ## Threshold update contract
//!
//! The shared session operation owns threshold updates; traversal only
//! invokes the stage. The policy retains the PA-aligned shape from
//! `search_l2_u8_q` (mean over the PQ's per-entry low-precision
//! distances × slack). The stage exposes:
//!
//! * `PrefilterSession::distance` for the threshold-mean recompute
//!   loop (scalar, called O(L) times when `pq_back_id` changes).
//! * `PrefilterSession::filter_compact` for the per-hop streaming
//!   compaction (drives `DistanceStream<DistanceKernelBackend>` with the
//!   prefetch + 4-way ILP shape we tuned earlier).
//!
//! ## Implementations
//!
//! * [`NoPrefilter`] (below) — zero-sized no-op.
//! * [`jl::JlPrefilter`] — JL Sparse 1024-bit Hamming filter (current
//!   `search_l2_u8_q` default).

pub mod jl;
pub mod jl_hadamard;
pub mod rabitq;

pub use jl::{JlMipsPrefilter, JlPrefilter};
pub use jl_hadamard::JlHadamardPrefilter;
pub use rabitq::RabitqPrefilter;

mod contract;
mod tuning;
pub(crate) use contract::{PrefilterOutcome, PrefilterSession, PrefilterStage};

/// An official prefilter stage.
pub type Prefilter<'a> = dyn PrefilterStage + 'a;

/// Zero-sized no-op prefilter. Use at dispatch sites where the cascade
/// runs admission directly off the graph neighbours.
///
/// The recipe always contains a stage, just like `NoRerank`. The zero-sized
/// session needs no heap allocation. Its complete filtering operation returns
/// a constant outcome without reading or writing scratch.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoPrefilter;

/// Session for [`NoPrefilter`] — all methods are no-ops.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NoPrefilterSession;

// SAFETY: shape describes the borrowed base; sessions own checked query storage
// and cannot outlive that base.
unsafe impl PrefilterStage for NoPrefilter {
    fn precision(
        &self,
    ) -> Option<crate::model::dataset::quantized_dataset::capabilities::Precision> {
        None
    }

    #[inline(always)]
    fn open<'a>(&'a self, _q: &[f32]) -> Box<dyn PrefilterSession + 'a> {
        Box::new(NoPrefilterSession)
    }
}

// SAFETY: streams retain input IDs, write only within the input-sized output
// region, and return the initialized compacted prefix length.
unsafe impl PrefilterSession for NoPrefilterSession {
    #[inline(always)]
    unsafe fn apply(
        &self,
        _scratch: &mut crate::model::scratch::InMemSearchScratch,
        _search_list_size: usize,
        _lookahead_lines: usize,
    ) -> PrefilterOutcome {
        PrefilterOutcome::default()
    }

    #[inline(always)]
    unsafe fn filter_compact(
        &self,
        _id_scratch: &mut Vec<u32>,
        _threshold: f32,
        _lookahead_lines: usize,
    ) {
        // A directly opened no-op kernel also preserves every candidate.
    }

    #[inline(always)]
    fn distance(&self, _vid: u32) -> f32 {
        0.0
    }
}

#[cfg(test)]
mod tests;
