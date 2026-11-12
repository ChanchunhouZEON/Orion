/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::{NoPrefilterSession, PrefilterSession};
use crate::{
    model::{Neighbor, scratch::InMemSearchScratch},
    search::diagnostics::SearchObserver,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
struct Observer {
    distances: (usize, usize),
    outcomes: Vec<(Option<f32>, Option<f32>, usize, usize)>,
}
impl SearchObserver for Observer {
    fn prefilter(&mut self, threshold: usize, candidates: usize) {
        self.distances.0 += threshold;
        self.distances.1 += candidates;
    }
    fn prefilter_outcome(
        &mut self,
        cutoff: Option<f32>,
        beam: Option<f32>,
        evaluated: usize,
        rejected: usize,
    ) {
        self.outcomes.push((cutoff, beam, evaluated, rejected));
    }
}
struct Kernel {
    threshold_calls: AtomicUsize,
}
// SAFETY: the fixture uses IDs 0..4 and owns no unchecked storage. Compaction
// only retains existing IDs, in order; the scalar distance is defined for any ID.
unsafe impl PrefilterSession for Kernel {
    fn distance(&self, id: u32) -> f32 {
        self.threshold_calls.fetch_add(1, Ordering::Relaxed);
        2.0 * (id + 1) as f32
    }
    unsafe fn filter_compact(&self, ids: &mut Vec<u32>, cutoff: f32, _: usize) {
        ids.retain(|&id| 2.0 * ((id + 1) as f32) < cutoff);
    }
}

#[test]
fn shared_step_preserves_cached_mean_history_and_work_accounting() {
    let kernel = Kernel {
        threshold_calls: AtomicUsize::new(0),
    };
    // Use the same object-safe session call as the traversal.
    let session: &dyn PrefilterSession = &kernel;
    let mut scratch = InMemSearchScratch::new(2);
    let mut observer = Observer::default();
    scratch.pq.insert(Neighbor::new(0, 10.0));
    scratch.id_scratch.extend([0, 1, 2]);
    unsafe {
        let outcome = session.apply(&mut scratch, 2, 32);
        let beam = (scratch.pq.size() >= 2).then(|| scratch.pq[scratch.pq.size() - 1].distance);
        outcome.observe(&mut observer, beam);
    }
    assert_eq!(scratch.id_scratch, [0, 1, 2]);
    assert_eq!(observer.outcomes, [(None, None, 0, 0)]);
    assert_eq!(scratch.prefilter_observations, 0);

    scratch.pq.insert(Neighbor::new(1, 20.0));
    unsafe {
        let outcome = session.apply(&mut scratch, 2, 32);
        let beam = (scratch.pq.size() >= 2).then(|| scratch.pq[scratch.pq.size() - 1].distance);
        outcome.observe(&mut observer, beam);
    }
    assert_eq!(scratch.id_scratch, [0]);
    assert_eq!(scratch.prefilter_frontier_mean, 3.0);
    assert_eq!(scratch.prefilter_frontier_mean_sum, 3.0);
    assert_eq!(observer.distances, (2, 3));
    let slack = super::tuning::prefilter_slack();
    assert_eq!(observer.outcomes[1], (Some(3.0 * slack), Some(20.0), 3, 2));

    scratch.id_scratch.extend([1, 2]);
    unsafe {
        let outcome = session.apply(&mut scratch, 2, 32);
        let beam = (scratch.pq.size() >= 2).then(|| scratch.pq[scratch.pq.size() - 1].distance);
        outcome.observe(&mut observer, beam);
    }
    assert_eq!(kernel.threshold_calls.load(Ordering::Relaxed), 2);
    assert_eq!(scratch.prefilter_frontier_mean_sum, 6.0);
    assert_eq!(scratch.prefilter_observations, 2);

    scratch.pq.clear();
    scratch.pq.insert(Neighbor::new(0, 10.0));
    scratch.pq.insert(Neighbor::new(2, 30.0));
    scratch.id_scratch.extend([1, 2]);
    unsafe {
        let outcome = session.apply(&mut scratch, 2, 32);
        let beam = (scratch.pq.size() >= 2).then(|| scratch.pq[scratch.pq.size() - 1].distance);
        outcome.observe(&mut observer, beam);
    }
    assert_eq!(kernel.threshold_calls.load(Ordering::Relaxed), 4);
    assert_eq!(scratch.prefilter_frontier_mean_sum, 10.0);
    assert_eq!(scratch.prefilter_observations, 3);
    assert_eq!(
        observer.outcomes[3],
        (Some((10.0 / 3.0) * slack), Some(30.0), 3, 2)
    );
    assert_eq!(observer.distances, (4, 9));
}

#[test]
fn no_op_preserves_candidates_and_history_without_counting_filter_work() {
    let mut scratch = InMemSearchScratch::new(1);
    scratch.pq.insert(Neighbor::new(0, 7.0));
    scratch.id_scratch.extend([3, 1, 2]);
    scratch.prefilter_frontier_mean_sum = 12.0;
    scratch.prefilter_observations = 3;
    scratch.prefilter_last_worst_id = 9;
    scratch.prefilter_frontier_mean = 4.0;
    let mut observer = Observer::default();
    unsafe {
        let session: &dyn PrefilterSession = &NoPrefilterSession;
        let outcome = session.apply(&mut scratch, 1, 32);
        assert_eq!(outcome, super::PrefilterOutcome::default());
        outcome.observe(&mut observer, Some(7.0));
    }
    assert_eq!(scratch.id_scratch, [3, 1, 2]);
    assert_eq!(scratch.prefilter_frontier_mean_sum, 12.0);
    assert_eq!(scratch.prefilter_observations, 3);
    assert_eq!(scratch.prefilter_last_worst_id, 9);
    assert_eq!(scratch.prefilter_frontier_mean, 4.0);
    assert_eq!(observer.distances, (0, 0));
    assert_eq!(observer.outcomes, [(None, Some(7.0), 0, 0)]);
}
