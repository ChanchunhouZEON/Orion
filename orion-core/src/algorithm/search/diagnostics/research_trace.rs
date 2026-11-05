/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::{Neighbor, SearchObserver};
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
