use super::SearchObserver;
use crate::model::Neighbor;

#[derive(Debug, Default, Clone, Copy)]
pub struct PhaseWork {
    pub expansions: usize,
    pub admission_ndc: usize,
    pub prefilter_candidate_ndc: usize,
    pub prefilter_threshold_ndc: usize,
}

impl PhaseWork {
    pub fn ndc(&self) -> usize {
        self.admission_ndc + self.prefilter_candidate_ndc + self.prefilter_threshold_ndc
    }
}

/// A permanent split at the first converged expansion, including later reversals
/// in the tail. Entry scoring belongs to the prefix; rerank is separate.
#[derive(Debug, Default)]
pub struct SynergyTrace {
    pub before: PhaseWork,
    pub after: PhaseWork,
    pub first_converged_step: Option<usize>,
    pub rerank_candidates: usize,
    pub early_exited: bool,
}

impl SynergyTrace {
    fn phase(&mut self) -> &mut PhaseWork {
        if self.first_converged_step.is_some() {
            &mut self.after
        } else {
            &mut self.before
        }
    }
}

impl SearchObserver for SynergyTrace {
    fn entry(&mut self, _id: u32) {
        self.before.admission_ndc += 1;
    }

    fn expansion(&mut self, converged: bool, _ids: &[u32]) {
        if converged && self.first_converged_step.is_none() {
            self.first_converged_step = Some(self.before.expansions + 1);
        }
        self.phase().expansions += 1;
    }

    fn prefilter(&mut self, threshold: usize, candidates: usize) {
        let phase = self.phase();
        phase.prefilter_threshold_ndc += threshold;
        phase.prefilter_candidate_ndc += candidates;
    }

    fn admission(&mut self, evaluations: usize, _admitted: &[Neighbor]) {
        self.phase().admission_ndc += evaluations;
    }

    fn rerank(&mut self, candidates: usize) {
        self.rerank_candidates = candidates;
    }

    fn early_exit(&mut self) {
        self.early_exited = true;
    }
}
