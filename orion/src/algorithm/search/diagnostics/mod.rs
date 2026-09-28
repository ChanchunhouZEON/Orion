//! Search diagnostics. Detailed traces are opt-in and are not QPS timings.
mod legacy;
mod synergy;
mod matched_state;
pub use matched_state::{Checkpoint, MatchedStateTrace};
pub use synergy::{PhaseWork, SynergyTrace};
pub use legacy::{SearchProfile, SearchProfileStats};

use crate::model::Neighbor;

/// Only the post-convergence neighbor collection changes between these arms.
/// Convergence updates and PQ flush scheduling remain identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborMode {
    FullNeighbor,
    LocalOnly,
    LocalExtra,
}

impl NeighborMode {
    pub const ALL: [Self; 3] = [Self::FullNeighbor, Self::LocalOnly, Self::LocalExtra];
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
    fn before_expansion(&mut self, _node: u32, _converged: bool,
        _scratch: &crate::model::scratch::InMemSearchScratch,
        _previous_admits: usize, _hops_since_flush: usize) {}
    fn after_flush(&mut self, _beam: &[Neighbor]) {}
    fn entry(&mut self, _id: u32) {}
    fn expansion(&mut self, _converged: bool, _ids: &[u32]) {}
    fn prefilter(&mut self, _threshold_evaluations: usize, _candidate_evaluations: usize) {}
    fn admission(&mut self, _evaluations: usize, _admitted: &[Neighbor]) {}
    fn rerank(&mut self, _candidates: usize) {}
    fn early_exit(&mut self) {}
}

pub struct NoopObserver;
impl SearchObserver for NoopObserver {}

/// Query-local counters; no shared/global aggregation is needed.
#[derive(Debug, Default)]
pub struct SearchTrace {
    pub expansions: usize,
    pub pre_expansions: usize,
    pub post_expansions: usize,
    pub first_switch_step: Option<usize>,
    pub reversals: usize,
    pub unique_candidates: usize,
    pub prefilter_threshold_ndc: usize,
    pub prefilter_candidate_ndc: usize,
    /// Includes the entry score, regardless of admission precision.
    pub admission_ndc: usize,
    /// Input size to the rerank stage; NoRerank performs no distance work.
    pub rerank_candidates: usize,
    pub targets: Vec<u32>,
    /// Step 0 denotes the entry; None means never encountered before prefilter.
    pub first_discovery: Vec<Option<usize>>,
    /// Passed the admission cutoff, not necessarily retained in the beam.
    pub first_admission: Vec<Option<usize>>,
    was_converged: bool,
}

impl SearchTrace {
    pub fn new(targets: &[u32]) -> Self {
        Self {
            targets: targets.to_vec(),
            first_discovery: vec![None; targets.len()],
            first_admission: vec![None; targets.len()],
            ..Self::default()
        }
    }

    fn mark(targets: &[u32], times: &mut [Option<usize>], id: u32, step: usize) {
        if let Some(pos) = targets.iter().position(|&target| target == id) {
            times[pos].get_or_insert(step);
        }
    }
}

impl SearchObserver for SearchTrace {
    fn entry(&mut self, id: u32) {
        self.unique_candidates += 1;
        self.admission_ndc += 1;
        Self::mark(&self.targets, &mut self.first_discovery, id, 0);
        Self::mark(&self.targets, &mut self.first_admission, id, 0);
    }

    fn expansion(&mut self, converged: bool, ids: &[u32]) {
        self.expansions += 1;
        if converged {
            self.post_expansions += 1;
            self.first_switch_step.get_or_insert(self.expansions);
        } else {
            self.pre_expansions += 1;
            if self.was_converged {
                self.reversals += 1;
            }
        }
        self.was_converged = converged;
        self.unique_candidates += ids.len();
        for &id in ids {
            Self::mark(
                &self.targets,
                &mut self.first_discovery,
                id,
                self.expansions,
            );
        }
    }

    fn prefilter(&mut self, threshold_evaluations: usize, candidate_evaluations: usize) {
        self.prefilter_threshold_ndc += threshold_evaluations;
        self.prefilter_candidate_ndc += candidate_evaluations;
    }

    fn admission(&mut self, evaluations: usize, admitted: &[Neighbor]) {
        self.admission_ndc += evaluations;
        for neighbor in admitted {
            Self::mark(
                &self.targets,
                &mut self.first_admission,
                neighbor.id,
                self.expansions,
            );
        }
    }

    fn rerank(&mut self, candidates: usize) {
        self.rerank_candidates = candidates;
    }
}
