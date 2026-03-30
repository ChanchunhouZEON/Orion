/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

/// Manages candidate sets gained from robust pruning process.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct CandidateSetManager {
    /// Mirrors diskann's member `candidate_anchor_set` for independent diskann-base construction.
    #[serde(skip, default)]
    pub candidate_anchor_sets:
        Option<Vec<Arc<Mutex<std::collections::HashMap<u32, HashSet<u32>>>>>>,

    /// candidate_sets[node_id] = HashSet of candidate node IDs
    /// Derived directly from NeighborPriorityQueue during greedy search
    pub candidate_sets: Vec<HashSet<u32>>,
}

impl CandidateSetManager {
    pub fn new(n: usize) -> Self {
        Self {
            candidate_anchor_sets: None,
            candidate_sets: vec![HashSet::new(); n],
        }
    }

    pub fn with_anchor_sets(n: usize) -> Self {
        let candidate_anchor_sets = Some(
            (0..n)
                .map(|_| Arc::new(Mutex::new(std::collections::HashMap::new())))
                .collect(),
        );

        Self {
            candidate_anchor_sets,
            candidate_sets: vec![HashSet::new(); n],
        }
    }

    /// Set the candidate set for a node (from NPQ contents).
    pub fn set_for_node(&mut self, node: u32, candidates: Vec<u32>) {
        self.candidate_sets[node as usize] = candidates.into_iter().collect();
    }

    /// Check if `candidate` is in `node`'s candidate set.
    pub fn contains(&self, node: u32, candidate: u32) -> bool {
        self.candidate_sets[node as usize].contains(&candidate)
    }

    pub fn get(&self, node: u32) -> &HashSet<u32> {
        &self.candidate_sets[node as usize]
    }

    pub fn candidate_sets_arc(&self) -> Arc<Vec<HashSet<u32>>> {
        Arc::new(self.candidate_sets.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_empty_sets() {
        let csm = CandidateSetManager::new(5);
        assert_eq!(csm.candidate_sets.len(), 5);
        for set in &csm.candidate_sets {
            assert!(set.is_empty());
        }
    }

    #[test]
    fn test_set_for_node() {
        let mut csm = CandidateSetManager::new(5);
        csm.set_for_node(0, vec![5, 1, 3, 0]);

        let set = csm.get(0);
        assert!(set.contains(&0));
        assert!(set.contains(&1));
        assert!(set.contains(&3));
        assert!(set.contains(&5));
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_contains() {
        let mut csm = CandidateSetManager::new(5);
        csm.set_for_node(2, vec![10, 5, 20, 2]);

        assert!(csm.contains(2, 2));
        assert!(csm.contains(2, 5));
        assert!(csm.contains(2, 10));
        assert!(csm.contains(2, 20));
        assert!(!csm.contains(2, 3));
        assert!(!csm.contains(2, 15));
    }

    #[test]
    fn test_candidate_sets_arc() {
        let mut csm = CandidateSetManager::new(3);
        csm.set_for_node(0, vec![1, 2]);
        csm.set_for_node(1, vec![0, 2]);

        let arc = csm.candidate_sets_arc();
        assert_eq!(arc.len(), 3);
        assert!(arc[0].contains(&1));
        assert!(arc[0].contains(&2));
        assert!(arc[1].contains(&0));
        assert!(arc[1].contains(&2));
        assert!(arc[2].is_empty());
    }

    #[test]
    fn test_empty_node_contains_nothing() {
        let csm = CandidateSetManager::new(3);
        assert!(!csm.contains(0, 0));
        assert!(!csm.contains(1, 5));
    }
}
