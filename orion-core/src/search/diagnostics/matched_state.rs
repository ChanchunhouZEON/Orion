/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::{SearchObserver, SearchTrace, SynergyTrace};
use crate::model::{Neighbor, search::scratch::InMemSearchScratch};
use std::collections::BTreeSet;

/// Logical state immediately after choosing the expanded vertex and updating DCC,
/// but before collecting neighbors. Distances are stored as bits for exact equality.
#[derive(Debug, PartialEq, Eq)]
pub struct Checkpoint {
    /// Internal node ID associated with this observation.
    pub node: u32,

    /// Expansion step at which the checkpoint was captured.
    pub step: usize,
    /// (id, distance[u32], visited) pairs come from current beam.
    pub beam: Vec<(u32, u32, bool)>,

    /// Next unexpanded beam ID, absent when the beam is exhausted.
    pub next_unvisited: Option<u32>,

    /// Beam capacity at the captured traversal state.
    pub capacity: usize,
    /// (id, distance[u32], visited) pairs come from pending `dist_buffer` which hasn't
    /// flushed into beam.
    pub pending: Vec<(u32, u32, bool)>,

    /// IDs already discovered at the checkpoint.
    pub seen: Vec<u32>,
    /// Record whether admission happened in each hop.
    pub convergence: Vec<u64>,
    /// (limit, consecutive_no_admit) mapped from `EarlyExitChecker`
    pub early_exit: (usize, usize),

    /// Bit-preserving snapshot of query-local prefilter counters and estimates.
    pub prefilter: (u32, u32, u32, u32),

    /// Admission count retained from the preceding traversal hop.
    pub previous_admits: usize,

    /// Traversal hops accumulated since pending candidates last entered the beam.
    pub hops_since_flush: usize,
}

impl Checkpoint {
    /// Copies the beam, pending candidates and adaptive state for exact checkpoint comparison.
    ///
    /// Float bit patterns are retained so diagnostics do not hide rounding differences.
    fn capture(
        node: u32,
        step: usize,
        s: &InMemSearchScratch,
        previous_admits: usize,
        hops_since_flush: usize,
    ) -> Self {
        let entries = |xs: &[Neighbor]| {
            xs.iter()
                .map(|n| (n.id, n.distance.to_bits(), n.visited))
                .collect()
        };
        Self {
            node,
            step,
            beam: entries(s.pq.neighbors()),
            next_unvisited: s.pq.peek_notvisited().map(|n| n.id),
            capacity: s.pq.capacity(),
            pending: entries(&s.dist_buffer),
            seen: s.seen.diagnostic_ids(),
            convergence: s.scc.diagnostic_state(),
            early_exit: s.early_exit.diagnostic_state(),
            prefilter: (
                s.prefilter_frontier_mean_sum.to_bits(),
                s.prefilter_observations,
                s.prefilter_last_worst_id,
                s.prefilter_frontier_mean.to_bits(),
            ),
            previous_admits,
            hops_since_flush,
        }
    }
}

/// Checkpoint and observations used to compare policies from the same search state.
pub struct MatchedStateTrace {
    /// First converged state captured for replay, absent before the switch.
    pub checkpoint: Option<Checkpoint>,

    /// Stage and phase work counters for the observed traversal.
    pub work: SynergyTrace,

    /// Target discovery/admission observations for the same traversal.
    pub discovery: SearchTrace,
    /// Every main-loop vertex expanded from the first checkpoint onward.
    pub tail_nodes: Vec<(u32, bool)>,
    /// IDs present at a real PQ flush after the first checkpoint.
    pub retained_after_switch: BTreeSet<u32>,

    /// Distinct IDs admitted after the captured convergence checkpoint.
    pub admitted_after_switch: BTreeSet<u32>,
}

impl MatchedStateTrace {
    /// Creates an empty matched-state trace for the selected target IDs.
    pub fn new(targets: &[u32]) -> Self {
        Self {
            checkpoint: None,
            work: SynergyTrace::default(),
            discovery: SearchTrace::new(targets),
            tail_nodes: Vec::new(),
            retained_after_switch: BTreeSet::new(),
            admitted_after_switch: BTreeSet::new(),
        }
    }
}

impl SearchObserver for MatchedStateTrace {
    /// Captures the first converged search state and records subsequent expansion order.
    fn before_expansion(
        &mut self,
        node: u32,
        converged: bool,
        scratch: &InMemSearchScratch,
        previous_admits: usize,
        hops_since_flush: usize,
    ) {
        if converged && self.checkpoint.is_none() {
            self.checkpoint = Some(Checkpoint::capture(
                node,
                self.discovery.expansions + 1,
                scratch,
                previous_admits,
                hops_since_flush,
            ));
        }
        if self.checkpoint.is_some() {
            self.tail_nodes.push((node, converged));
        }
    }

    /// Forwards the entry event to work and target-discovery accounting.
    fn entry(&mut self, id: u32) {
        self.work.entry(id);
        self.discovery.entry(id);
    }

    /// Forwards expansion observations without modifying the beam.
    fn expansion(&mut self, converged: bool, ids: &[u32]) {
        self.work.expansion(converged, ids);
        self.discovery.expansion(converged, ids);
    }

    /// Forwards prefilter work to the paired diagnostic collectors.
    fn prefilter(&mut self, threshold: usize, candidates: usize) {
        self.work.prefilter(threshold, candidates);
        self.discovery.prefilter(threshold, candidates);
    }

    /// Tracks admission work and records IDs admitted after the saved checkpoint.
    fn admission(&mut self, evaluations: usize, admitted: &[Neighbor]) {
        self.work.admission(evaluations, admitted);
        self.discovery.admission(evaluations, admitted);
        if self.checkpoint.is_some() {
            self.admitted_after_switch
                .extend(admitted.iter().map(|n| n.id));
        }
    }

    /// Records beam IDs retained after each post-checkpoint flush.
    fn after_flush(&mut self, beam: &[Neighbor]) {
        if self.checkpoint.is_some() {
            self.retained_after_switch.extend(beam.iter().map(|n| n.id));
        }
    }

    /// Forwards the final rerank count to both diagnostic collectors.
    fn rerank(&mut self, n: usize) {
        self.work.rerank(n);
        self.discovery.rerank(n);
    }

    /// Marks early termination in the work trace.
    fn early_exit(&mut self) {
        self.work.early_exit();
    }
}

// Keep diagnostics bounded: do not format vector payloads or working buffers.
impl std::fmt::Debug for MatchedStateTrace {
    /// Formats `has_checkpoint`, `tail_nodes`, `retained_after_switch`, `admitted_after_switch` for diagnostic inspection.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatchedStateTrace")
            .field("has_checkpoint", &self.checkpoint.is_some())
            .field("tail_nodes", &self.tail_nodes.len())
            .field("retained_after_switch", &self.retained_after_switch.len())
            .field("admitted_after_switch", &self.admitted_after_switch.len())
            .finish_non_exhaustive()
    }
}
