/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::SearchObserver;
use crate::model::Neighbor;

/// Distance and expansion counts for one adaptive-search phase.
#[derive(Debug, Default, Clone, Copy)]
pub struct PhaseWork {
    /// Number of graph node expansions observed.
    pub expansions: usize,

    /// Admission distance evaluations, including the entry evaluation where applicable.
    pub admission_ndc: usize,

    /// Candidate-distance evaluations performed by the prefilter.
    pub prefilter_candidate_ndc: usize,

    /// Distance evaluations used to establish prefilter thresholds.
    pub prefilter_threshold_ndc: usize,
}

impl PhaseWork {
    /// Sums admission and filter distance work for this phase, including threshold construction.
    pub fn ndc(&self) -> usize {
        self.admission_ndc + self.prefilter_candidate_ndc + self.prefilter_threshold_ndc
    }
}

/// A permanent split at the first converged expansion, including later reversals
/// in the tail. Entry scoring belongs to the prefix; rerank is separate.
#[derive(Debug, Default)]
pub struct SynergyTrace {
    /// Work observed before the adaptive convergence switch.
    pub before: PhaseWork,

    /// Work observed after the adaptive convergence switch.
    pub after: PhaseWork,

    /// First converged expansion step, absent if convergence never occurred.
    pub first_converged_step: Option<usize>,

    /// Number of candidates passed to final reranking.
    pub rerank_candidates: usize,

    /// Whether the adaptive early-exit rule stopped traversal.
    pub early_exited: bool,
}

impl SynergyTrace {
    /// Selects the work counters before or after the first observed convergence transition.
    fn phase(&mut self) -> &mut PhaseWork {
        if self.first_converged_step.is_some() {
            &mut self.after
        } else {
            &mut self.before
        }
    }
}

impl SearchObserver for SynergyTrace {
    /// Counts the entry evaluation in the pre-convergence admission work.
    fn entry(&mut self, _id: u32) {
        self.before.admission_ndc += 1;
    }

    /// Records the first converged step and increments work in the active phase.
    fn expansion(&mut self, converged: bool, _ids: &[u32]) {
        if converged && self.first_converged_step.is_none() {
            self.first_converged_step = Some(self.before.expansions + 1);
        }
        self.phase().expansions += 1;
    }

    /// Accumulates threshold and candidate prefilter evaluations in the active phase.
    fn prefilter(&mut self, threshold: usize, candidates: usize) {
        let phase = self.phase();
        phase.prefilter_threshold_ndc += threshold;
        phase.prefilter_candidate_ndc += candidates;
    }

    /// Accumulates admission distance evaluations in the active phase.
    fn admission(&mut self, evaluations: usize, _admitted: &[Neighbor]) {
        self.phase().admission_ndc += evaluations;
    }

    /// Records the final rerank candidate count.
    fn rerank(&mut self, candidates: usize) {
        self.rerank_candidates = candidates;
    }

    /// Marks that the adaptive stopping rule terminated this search.
    fn early_exit(&mut self) {
        self.early_exited = true;
    }
}
