/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Search diagnostics. Detailed traces are opt-in and are not QPS timings.
#[cfg(feature = "benchmark")]
mod legacy;
#[cfg(feature = "benchmark")]
mod matched_state;
pub(crate) mod metrics;
#[cfg(feature = "benchmark")]
mod synergy;
#[cfg(feature = "benchmark")]
pub use legacy::{SearchProfile, SearchProfileStats};
#[cfg(feature = "benchmark")]
pub use matched_state::{Checkpoint, MatchedStateTrace};
#[cfg(feature = "benchmark")]
pub use synergy::{PhaseWork, SynergyTrace};

use crate::model::Neighbor;

/// Only the post-convergence neighbor collection changes between these arms.
/// Convergence updates and PQ flush scheduling remain identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborMode {
    FullNeighbor,
    LocalOnly,
    LocalExtra,
}

impl Default for NeighborMode {
    /// Selects production local-plus-extra traversal unless the benchmark override disables extras.
    fn default() -> Self {
        if super::include_extras() {
            Self::LocalExtra
        } else {
            Self::FullNeighbor
        }
    }
}

impl NeighborMode {
    #[cfg(feature = "benchmark")]
    pub const ALL: [Self; 3] = [Self::FullNeighbor, Self::LocalOnly, Self::LocalExtra];

    /// Returns the stable benchmark label for a neighbor-selection policy.
    #[cfg(feature = "benchmark")]
    pub fn name(self) -> &'static str {
        match self {
            Self::FullNeighbor => "full_neighbor",
            Self::LocalOnly => "local_only",
            Self::LocalExtra => "local_extra",
        }
    }
}

/// Hooks receive only visited-set-deduplicated candidates.
/// A no-op observer is monomorphized away in the normal search path.
pub trait SearchObserver {
    /// Hooks dedicated to portable traces, including the normally unobserved peel.
    /// Empty defaults preserve existing research observers and compile away.
    fn scored_entry(&mut self, _id: u32, _distance: f32) {}

    /// Observes which node begins expansion and whether it is in the peel or converged phase.
    fn expansion_started(&mut self, _node: u32, _converged: bool, _peel: bool) {}

    /// Borrows the current beam after an update; copy only the state needed by the observer.
    fn beam_updated(&mut self, _beam: &[Neighbor]) {}

    /// Observes the complete query scratch state before expanding a node for matched-state diagnostics.
    fn before_expansion(
        &mut self,
        _node: u32,
        _converged: bool,
        _scratch: &crate::model::search::scratch::InMemSearchScratch,
        _previous_admits: usize,
        _hops_since_flush: usize,
    ) {
    }

    /// Observes the beam after buffered admissions have been merged.
    fn after_flush(&mut self, _beam: &[Neighbor]) {}

    /// Records the initial graph node selected for this query.
    fn entry(&mut self, _id: u32) {}

    /// Observes deduplicated candidate IDs selected for the current expansion phase.
    fn expansion(&mut self, _converged: bool, _ids: &[u32]) {}

    /// Accounts separately for threshold-building distances and candidate filter distances.
    fn prefilter(&mut self, _threshold_evaluations: usize, _candidate_evaluations: usize) {}
    /// Actual per-expansion filter outcome; absent cutoff means the stage was bypassed.
    fn prefilter_outcome(
        &mut self,
        _cutoff: Option<f32>,
        _admission_cutoff: Option<f32>,
        _tested: usize,
        _rejected: usize,
    ) {
    }

    /// Observes scored admissions and the number of candidate evaluations that produced them.
    fn admission(&mut self, _evaluations: usize, _admitted: &[Neighbor]) {}

    /// Records the number of candidates submitted to final reranking.
    fn rerank(&mut self, _candidates: usize) {}

    /// Records termination caused by the adaptive no-admission rule.
    fn early_exit(&mut self) {}
}

/// Observer that records nothing, allowing ordinary searches to omit tracing work.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopObserver;
impl SearchObserver for NoopObserver {}

/// Query-local counters; no shared/global aggregation is needed.
#[cfg(feature = "benchmark")]
mod research_trace;
#[cfg(feature = "benchmark")]
pub use research_trace::SearchTrace;

/// Offline neighbor contribution analysis; never part of production search.
#[cfg(feature = "benchmark")]
pub mod analysis;
