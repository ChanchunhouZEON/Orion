/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::{Neighbor, SearchObserver};

/// Target-discovery and stage-work observations collected without changing traversal.
#[derive(Debug, Default)]
pub struct SearchTrace {
    /// Number of graph node expansions observed.
    pub expansions: usize,

    /// Expansions performed before convergence.
    pub pre_expansions: usize,

    /// Expansions performed in the converged phase.
    pub post_expansions: usize,

    /// First observed transition to the converged phase.
    pub first_switch_step: Option<usize>,

    /// Transitions from converged back to unconverged traversal.
    pub reversals: usize,

    /// Entry plus unseen candidates exposed by expansion events.
    pub unique_candidates: usize,

    /// Distance evaluations used to establish prefilter thresholds.
    pub prefilter_threshold_ndc: usize,

    /// Candidate-distance evaluations performed by the prefilter.
    pub prefilter_candidate_ndc: usize,
    /// Includes the entry score, regardless of admission precision.
    pub admission_ndc: usize,
    /// Input size to the rerank stage; NoRerank performs no distance work.
    pub rerank_candidates: usize,

    /// Reference neighbor IDs whose discovery and admission are tracked.
    pub targets: Vec<u32>,
    /// Step 0 denotes the entry; None means never encountered before prefilter.
    pub first_discovery: Vec<Option<usize>>,
    /// Passed the admission cutoff, not necessarily retained in the beam.
    pub first_admission: Vec<Option<usize>>,
    was_converged: bool,
}

impl SearchTrace {
    /// Tracks first discovery and admission of selected ground-truth IDs during one query.
    pub fn new(targets: &[u32]) -> Self {
        Self {
            targets: targets.to_vec(),
            first_discovery: vec![None; targets.len()],
            first_admission: vec![None; targets.len()],
            ..Self::default()
        }
    }

    /// Records the first step at which a target ID is observed, leaving earlier observations intact.
    fn mark(targets: &[u32], times: &mut [Option<usize>], id: u32, step: usize) {
        if let Some(pos) = targets.iter().position(|&target| target == id) {
            times[pos].get_or_insert(step);
        }
    }
}

impl SearchObserver for SearchTrace {
    /// Records entry discovery and admission at step zero.
    fn entry(&mut self, id: u32) {
        self.unique_candidates += 1;
        self.admission_ndc += 1;
        Self::mark(&self.targets, &mut self.first_discovery, id, 0);
        Self::mark(&self.targets, &mut self.first_admission, id, 0);
    }

    /// Tracks phase transitions and the first discovery step of target neighbors.
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

    /// Counts threshold and candidate prefilter evaluations separately.
    fn prefilter(&mut self, threshold_evaluations: usize, candidate_evaluations: usize) {
        self.prefilter_threshold_ndc += threshold_evaluations;
        self.prefilter_candidate_ndc += candidate_evaluations;
    }

    /// Counts distance work and records when target neighbors first enter the beam.
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

    /// Records how many candidates reached final reranking.
    fn rerank(&mut self, candidates: usize) {
        self.rerank_candidates = candidates;
    }
}
