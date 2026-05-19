/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use diskann::model::{Neighbor, NeighborPriorityQueue};
use hashbrown::HashSet;

/// Minimal per-thread scratch for baseline beam search.
///
/// Kept standalone (rather than reusing `diskann::model::InMemQueryScratch`)
/// because the full DiskANN scratch carries build-time buffers (occlude-list
/// output, pq scratch, etc.) that are unused during plain search and require
/// IndexWriteParameters at construction time. Baselines only need a priority
/// queue, a visited set, and an ID staging buffer.
pub struct BaselineScratch {
    pub best_candidates: NeighborPriorityQueue,
    pub visited: HashSet<u32>,
    pub id_buffer: Vec<u32>,
}

impl BaselineScratch {
    /// Create a scratch sized for a given search list and node count.
    pub fn new(search_list_size: usize, num_nodes: usize) -> Self {
        Self {
            best_candidates: NeighborPriorityQueue::with_capacity(search_list_size),
            visited: HashSet::with_capacity(num_nodes.min(20 * search_list_size.max(1))),
            id_buffer: Vec::with_capacity(128),
        }
    }

    /// Reset state for a new query, keeping allocations. If `search_list_size`
    /// grew since the last call, the priority queue is expanded.
    pub fn prepare(&mut self, search_list_size: usize) {
        self.best_candidates.clear();
        self.best_candidates.reserve(search_list_size);
        self.best_candidates.set_capacity(search_list_size);
        self.visited.clear();
        self.id_buffer.clear();
    }

    /// Seed the priority queue with the entry point.
    #[inline]
    pub fn seed_entry(&mut self, entry: u32, entry_dist: f32) {
        self.visited.insert(entry);
        self.best_candidates
            .insert(Neighbor::new(entry, entry_dist));
    }
}
