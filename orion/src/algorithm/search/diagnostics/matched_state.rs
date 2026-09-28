use super::{SearchObserver, SearchTrace, SynergyTrace};
use crate::model::{scratch::InMemSearchScratch, Neighbor};
use std::collections::BTreeSet;

/// Logical state immediately after choosing the expanded vertex and updating DCC,
/// but before collecting neighbors. Distances are stored as bits for exact equality.
#[derive(Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub node: u32,
    pub step: usize,
    /// (id, distance[u32], visited) pairs come from current beam.
    pub beam: Vec<(u32, u32, bool)>,
    pub next_unvisited: Option<u32>,
    pub capacity: usize,
    /// (id, distance[u32], visited) pairs come from pending `dist_buffer` which hasn't
    /// flushed into beam.
    pub pending: Vec<(u32, u32, bool)>,
    pub seen: Vec<u32>,
    /// Record whether admission happened in each hop.
    pub convergence: Vec<u64>,
    /// (limit, consecutive_no_admit) mapped from [EarlyExitChecker]
    pub early_exit: (usize, usize),
    pub prefilter: (u32, u32, u32, u32),
    pub previous_admits: usize,
    pub hops_since_flush: usize,
}

impl Checkpoint {
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
                s.jl_threshold_sum.to_bits(),
                s.jl_threshold_count,
                s.jl_last_worst_id,
                s.jl_tail_mean.to_bits(),
            ),
            previous_admits,
            hops_since_flush,
        }
    }
}

pub struct MatchedStateTrace {
    pub checkpoint: Option<Checkpoint>,
    pub work: SynergyTrace,
    pub discovery: SearchTrace,
    /// Every main-loop vertex expanded from the first checkpoint onward.
    pub tail_nodes: Vec<(u32, bool)>,
    /// IDs present at a real PQ flush after the first checkpoint.
    pub retained_after_switch: BTreeSet<u32>,
    pub admitted_after_switch: BTreeSet<u32>,
}

impl MatchedStateTrace {
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
    fn entry(&mut self, id: u32) {
        self.work.entry(id);
        self.discovery.entry(id);
    }
    fn expansion(&mut self, converged: bool, ids: &[u32]) {
        self.work.expansion(converged, ids);
        self.discovery.expansion(converged, ids);
    }
    fn prefilter(&mut self, threshold: usize, candidates: usize) {
        self.work.prefilter(threshold, candidates);
        self.discovery.prefilter(threshold, candidates);
    }
    fn admission(&mut self, evaluations: usize, admitted: &[Neighbor]) {
        self.work.admission(evaluations, admitted);
        self.discovery.admission(evaluations, admitted);
        if self.checkpoint.is_some() {
            self.admitted_after_switch
                .extend(admitted.iter().map(|n| n.id));
        }
    }
    fn after_flush(&mut self, beam: &[Neighbor]) {
        if self.checkpoint.is_some() {
            self.retained_after_switch.extend(beam.iter().map(|n| n.id));
        }
    }
    fn rerank(&mut self, n: usize) {
        self.work.rerank(n);
        self.discovery.rerank(n);
    }
    fn early_exit(&mut self) {
        self.work.early_exit();
    }
}
