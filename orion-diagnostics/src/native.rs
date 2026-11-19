/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Adapter from the production observer hooks to the portable trace schema.
use crate::{Candidate, Phase, SearchTrace, TraceEvent, TraceLimits, TraceMetadata, TraceRecorder};
use orion_ann_core::model::Neighbor;
use orion_ann_core::search::diagnostics::SearchObserver;

/// Converts native traversal callbacks into a bounded portable trace.
pub struct RecordingObserver {
    recorder: TraceRecorder,
    node: u32,
    step: u32,
    phase: Phase,
}
impl RecordingObserver {
    /// Creates a recorder after validating search metadata and recording limits.
    pub fn new(metadata: TraceMetadata, limits: TraceLimits) -> crate::Result<Self> {
        Ok(Self {
            recorder: TraceRecorder::new(metadata, limits)?,
            node: 0,
            step: 0,
            phase: Phase::Peel,
        })
    }

    /// Attaches final internal result IDs and validates the recorded search trace.
    pub fn finish(self, result_ids: Vec<u32>) -> crate::Result<SearchTrace> {
        self.recorder.finish(result_ids)
    }
}

/// Copies a neighbor into the portable score representation, preserving non-finite values.
fn candidate(n: &Neighbor) -> Candidate {
    Candidate {
        id: n.id,
        distance: n.distance.into(),
    }
}
impl SearchObserver for RecordingObserver {
    /// Records the scored entry node before traversal begins.
    fn scored_entry(&mut self, id: u32, distance: f32) {
        self.recorder.record(1, || TraceEvent::Entry {
            candidate: Candidate {
                id,
                distance: distance.into(),
            },
        });
    }

    /// Remembers the source node and traversal phase for the next expansion event.
    fn expansion_started(&mut self, node: u32, converged: bool, peel: bool) {
        self.node = node;
        self.phase = if peel {
            Phase::Peel
        } else if converged {
            Phase::Converged
        } else {
            Phase::Exploration
        };
    }

    /// Records the deduplicated expansion IDs and advances the expansion step counter.
    fn expansion(&mut self, _: bool, ids: &[u32]) {
        self.step += 1;
        self.recorder.record(ids.len(), || TraceEvent::Expansion {
            node: self.node,
            step: self.step,
            phase: self.phase,
            candidates: ids.to_vec(),
        });
    }

    /// Records both cutoff domains and rejection counts without copying candidate vectors.
    fn prefilter_outcome(
        &mut self,
        cutoff: Option<f32>,
        admission_cutoff: Option<f32>,
        tested: usize,
        rejected: usize,
    ) {
        self.recorder.record(0, || TraceEvent::Prefilter {
            cutoff: cutoff.map(Into::into),
            admission_cutoff: admission_cutoff.map(Into::into),
            tested: tested as u32,
            rejected: rejected as u32,
        });
    }

    /// Copies admitted candidates only while the recorder still has payload budget.
    fn admission(&mut self, evaluations: usize, admitted: &[Neighbor]) {
        self.recorder
            .record(admitted.len(), || TraceEvent::Admission {
                evaluated: evaluations as u32,
                admitted: admitted.iter().map(candidate).collect(),
            });
    }

    /// Captures the current beam in order while recording capacity remains.
    fn beam_updated(&mut self, beam: &[Neighbor]) {
        self.recorder.record(beam.len(), || TraceEvent::Beam {
            candidates: beam.iter().map(candidate).collect(),
        });
    }

    /// Records how many candidates reached the final reranking stage.
    fn rerank(&mut self, candidates: usize) {
        self.recorder.record(0, || TraceEvent::Rerank {
            candidates: candidates as u32,
        });
    }

    /// Propagates the search termination reason independently of recording truncation.
    fn early_exit(&mut self) {
        self.recorder.early_exit();
    }
}
