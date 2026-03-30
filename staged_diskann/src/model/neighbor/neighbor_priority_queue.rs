/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::model::Neighbor;
use std::collections::HashSet;

/// Sorted priority queue for neighbor search, ordered by ascending distance.
#[derive(Debug)]
pub struct NeighborPriorityQueue {
    size: usize,
    capacity: usize,
    cur: usize,
    data: Vec<Neighbor>,
    /// Track inserted IDs for O(1) dedup (binary search can miss duplicates
    /// when the same ID exists at a different distance position).
    ids: HashSet<u32>,
}

impl Default for NeighborPriorityQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl NeighborPriorityQueue {
    pub fn new() -> Self {
        Self {
            size: 0,
            capacity: 0,
            cur: 0,
            data: Vec::new(),
            ids: HashSet::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            size: 0,
            capacity,
            cur: 0,
            data: vec![Neighbor::default(); capacity + 1],
            ids: HashSet::with_capacity(capacity + 1),
        }
    }

    /// Insert a neighbor maintaining sorted order by distance.
    /// Returns the evicted neighbor's id if the queue was full and an item was displaced.
    pub fn insert(&mut self, nbr: Neighbor) -> Option<u32> {
        // O(1) dedup via HashSet — binary search alone can miss duplicates
        // when the same ID exists at a different distance position.
        if self.ids.contains(&nbr.id) {
            return None;
        }

        if self.size == self.capacity && self.data[self.size - 1] < nbr {
            return None;
        }

        let mut lo = 0;
        let mut hi = self.size;
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if nbr < self.data[mid] {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        // Track the evicted element (if any) before shifting
        let evicted = if self.size == self.capacity {
            let evicted_id = self.data[self.size - 1].id;
            self.ids.remove(&evicted_id);
            Some(evicted_id)
        } else {
            None
        };

        if lo < self.capacity {
            self.data.copy_within(lo..self.size, lo + 1);
        }
        self.data[lo] = Neighbor::new(nbr.id, nbr.distance);
        self.ids.insert(nbr.id);
        if self.size < self.capacity {
            self.size += 1;
        }
        if lo < self.cur {
            self.cur = lo;
        }

        evicted
    }

    /// Get the closest unvisited neighbor and mark it as visited.
    pub fn closest_notvisited(&mut self) -> Neighbor {
        self.data[self.cur].visited = true;
        let pre = self.cur;
        while self.cur < self.size && self.data[self.cur].visited {
            self.cur += 1;
        }
        self.data[pre]
    }

    pub fn has_notvisited_node(&self) -> bool {
        self.cur < self.size
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        if capacity < self.data.len() {
            self.capacity = capacity;
        }
    }

    pub fn reserve(&mut self, capacity: usize) {
        if capacity > self.capacity {
            self.data.resize(capacity + 1, Neighbor::default());
            self.capacity = capacity;
        }
    }

    pub fn clear(&mut self) {
        self.size = 0;
        self.cur = 0;
        self.ids.clear();
    }

    pub fn neighbors(&self) -> &[Neighbor] {
        &self.data[..self.size]
    }
}

impl std::ops::Index<usize> for NeighborPriorityQueue {
    type Output = Neighbor;

    fn index(&self, i: usize) -> &Self::Output {
        &self.data[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_maintains_sorted_order() {
        let mut pq = NeighborPriorityQueue::with_capacity(5);
        pq.insert(Neighbor::new(0, 3.0));
        pq.insert(Neighbor::new(1, 1.0));
        pq.insert(Neighbor::new(2, 2.0));

        assert_eq!(pq.size(), 3);
        assert_eq!(pq[0].id, 1); // closest first
        assert_eq!(pq[1].id, 2);
        assert_eq!(pq[2].id, 0);
    }

    #[test]
    fn test_insert_capacity_eviction() {
        let mut pq = NeighborPriorityQueue::with_capacity(2);
        pq.insert(Neighbor::new(0, 1.0));
        pq.insert(Neighbor::new(1, 2.0));
        pq.insert(Neighbor::new(2, 1.5)); // should evict id=1 (dist=2.0)

        assert_eq!(pq.size(), 2);
        assert_eq!(pq[0].id, 0);
        assert_eq!(pq[1].id, 2);
    }

    #[test]
    fn test_insert_reject_when_full_and_farther() {
        let mut pq = NeighborPriorityQueue::with_capacity(2);
        pq.insert(Neighbor::new(0, 1.0));
        pq.insert(Neighbor::new(1, 2.0));
        let evicted = pq.insert(Neighbor::new(2, 5.0)); // too far, rejected

        assert!(evicted.is_none());
        assert_eq!(pq.size(), 2);
    }

    #[test]
    fn test_insert_dedup_same_distance() {
        let mut pq = NeighborPriorityQueue::with_capacity(5);
        pq.insert(Neighbor::new(0, 1.0));
        let evicted = pq.insert(Neighbor::new(0, 1.0)); // same id, same distance

        assert!(evicted.is_none());
        assert_eq!(pq.size(), 1); // deduplicated
    }

    #[test]
    fn test_closest_notvisited() {
        let mut pq = NeighborPriorityQueue::with_capacity(3);
        pq.insert(Neighbor::new(0, 1.0));
        pq.insert(Neighbor::new(1, 2.0));
        pq.insert(Neighbor::new(2, 3.0));

        let first = pq.closest_notvisited();
        assert_eq!(first.id, 0);
        assert!(pq.has_notvisited_node());

        let second = pq.closest_notvisited();
        assert_eq!(second.id, 1);
    }

    #[test]
    fn test_all_visited() {
        let mut pq = NeighborPriorityQueue::with_capacity(2);
        pq.insert(Neighbor::new(0, 1.0));
        pq.insert(Neighbor::new(1, 2.0));

        pq.closest_notvisited();
        pq.closest_notvisited();
        assert!(!pq.has_notvisited_node());
    }

    #[test]
    fn test_clear() {
        let mut pq = NeighborPriorityQueue::with_capacity(5);
        pq.insert(Neighbor::new(0, 1.0));
        pq.insert(Neighbor::new(1, 2.0));
        pq.clear();
        assert_eq!(pq.size(), 0);
        assert!(!pq.has_notvisited_node());
    }

    #[test]
    fn test_neighbors_slice() {
        let mut pq = NeighborPriorityQueue::with_capacity(5);
        pq.insert(Neighbor::new(0, 1.0));
        pq.insert(Neighbor::new(1, 2.0));
        let nbrs = pq.neighbors();
        assert_eq!(nbrs.len(), 2);
    }

    #[test]
    fn test_dedup_same_id_different_distance() {
        // Regression: binary search could miss a duplicate when the same ID
        // exists at a very different distance position.
        let mut pq = NeighborPriorityQueue::with_capacity(10);
        pq.insert(Neighbor::new(1, 1.0));
        pq.insert(Neighbor::new(2, 2.0));
        pq.insert(Neighbor::new(5, 3.0));
        pq.insert(Neighbor::new(3, 3.5));
        pq.insert(Neighbor::new(7, 5.0));

        // Try to insert id=5 again with a very different distance
        let evicted = pq.insert(Neighbor::new(5, 100.0));
        assert!(evicted.is_none());
        assert_eq!(pq.size(), 5); // must not grow

        // Also try inserting with a smaller distance
        let evicted = pq.insert(Neighbor::new(5, 0.1));
        assert!(evicted.is_none());
        assert_eq!(pq.size(), 5);
    }

    #[test]
    fn test_eviction_updates_id_set() {
        let mut pq = NeighborPriorityQueue::with_capacity(3);
        pq.insert(Neighbor::new(0, 1.0));
        pq.insert(Neighbor::new(1, 2.0));
        pq.insert(Neighbor::new(2, 3.0)); // full at capacity 3

        // Insert better: evicts id=2
        pq.insert(Neighbor::new(3, 1.5));
        assert_eq!(pq.size(), 3);

        // Now id=2 was evicted, we should be able to re-insert it
        pq.insert(Neighbor::new(2, 1.2));
        assert_eq!(pq.size(), 3);
        // id=2 should be at position 1 (after id=0 at dist 1.0)
        assert_eq!(pq[1].id, 2);
    }
}
