/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::model::Neighbor;

/// Distance-ordered candidate storage with an expansion cursor and bounded beam.
#[derive(Debug)]
pub struct NeighborPriorityQueue {
    size: usize,
    capacity: usize,
    cur: usize,
    data: Vec<Neighbor>,
}

impl Default for NeighborPriorityQueue {
    /// Creates an empty beam with no reserved candidate capacity.
    fn default() -> Self {
        Self::new()
    }
}

impl NeighborPriorityQueue {
    /// Creates an empty queue; reserve positive beam capacity before inserting candidates.
    pub fn new() -> Self {
        Self {
            size: 0,
            capacity: 0,
            cur: 0,
            data: Vec::new(),
        }
    }

    /// Allocates a bounded beam and its extra insertion slot.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            size: 0,
            capacity,
            cur: 0,
            data: vec![Neighbor::default(); capacity + 1],
        }
    }

    /// Inserts a competitive candidate in sorted order and updates the next unvisited position.
    ///
    /// The queue must have positive capacity and a valid insertion slot.
    pub fn insert(&mut self, nbr: Neighbor) {
        if self.size == self.capacity && self.get_at(self.size - 1) < &nbr {
            return;
        }

        let mut lo = 0;
        let mut hi = self.size;
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if &nbr < self.get_at(mid) {
                hi = mid;
            } else if self.get_at(mid).id == nbr.id {
                return;
            } else {
                lo = mid + 1;
            }
        }

        if lo < self.capacity {
            self.data.copy_within(lo..self.size, lo + 1);
        }
        // Preserve caller-set `visited` so pre-expanded buffer entries stay
        // marked when inserted into the queue.
        self.data[lo] = nbr;
        if self.size < self.capacity {
            self.size += 1;
        }
        // Only rewind `cur` if the inserted element itself is not already
        // visited — otherwise the cur should skip over it.
        if lo < self.cur && !nbr.visited {
            self.cur = lo;
        }
    }

    /// Borrows an initialized queue slot without bounds checks; callers maintain the storage index invariant.
    fn get_at(&self, index: usize) -> &Neighbor {
        unsafe { self.data.get_unchecked(index) }
    }

    /// Marks and returns the next unvisited candidate; call only after checking availability.
    pub fn closest_notvisited(&mut self) -> Neighbor {
        self.data[self.cur].visited = true;
        let pre = self.cur;
        while self.cur < self.size && self.get_at(self.cur).visited {
            self.cur += 1;
        }
        self.data[pre]
    }

    /// Reports whether traversal still has an unexpanded candidate in the beam.
    pub fn has_notvisited_node(&self) -> bool {
        self.cur < self.size
    }

    /// Peek at the closest not-yet-visited node without marking it visited.
    /// Returns `None` when all nodes have been visited.
    pub fn peek_notvisited(&self) -> Option<&Neighbor> {
        if self.cur < self.size {
            Some(&self.data[self.cur])
        } else {
            None
        }
    }

    /// Returns the number of live candidates, excluding the insertion slot.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the configured beam bound, not the allocated vector length.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Changes the beam bound, growing storage when needed; callers clear state before shrinking a live beam.
    pub fn set_capacity(&mut self, capacity: usize) {
        if capacity > self.data.len() {
            self.data.resize(capacity + 1, Neighbor::default());
        }
        self.capacity = capacity;
    }

    /// Grows beam storage and its logical capacity only when the requested bound is larger.
    pub fn reserve(&mut self, capacity: usize) {
        if capacity > self.capacity {
            self.data.resize(capacity + 1, Neighbor::default());
            self.capacity = capacity;
        }
    }

    /// Resets beam length and visit cursor while keeping allocated storage for reuse.
    pub fn clear(&mut self) {
        self.size = 0;
        self.cur = 0;
    }
}

impl std::ops::Index<usize> for NeighborPriorityQueue {
    type Output = Neighbor;

    /// Borrows the stored candidate at a physical queue position.
    ///
    /// # Panics
    ///
    /// Panics when the position is outside the backing vector.
    fn index(&self, i: usize) -> &Self::Output {
        &self.data[i]
    }
}

#[cfg(test)]
mod neighbor_priority_queue_test {
    use super::*;

    #[test]
    fn test_insert() {
        let mut queue = NeighborPriorityQueue::with_capacity(3);
        assert_eq!(queue.size(), 0);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        assert_eq!(queue.size(), 2);
        queue.insert(Neighbor::new(2, 0.5));
        assert_eq!(queue.size(), 2);
        queue.insert(Neighbor::new(3, 0.9));
        assert_eq!(queue.size(), 3);
        assert_eq!(queue[2].id, 1);
        queue.insert(Neighbor::new(4, 2.0));
        assert_eq!(queue.size(), 3);
        assert_eq!(queue[0].id, 2);
        assert_eq!(queue[1].id, 3);
        assert_eq!(queue[2].id, 1);
    }

    #[test]
    fn test_visit() {
        let mut queue = NeighborPriorityQueue::with_capacity(3);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        assert!(queue.has_notvisited_node());
        let nbr = queue.closest_notvisited();
        assert_eq!(nbr.id, 2);
        let nbr = queue.closest_notvisited();
        assert_eq!(nbr.id, 1);
        let nbr = queue.closest_notvisited();
        assert_eq!(nbr.id, 3);
        assert!(!queue.has_notvisited_node());
    }

    #[test]
    fn test_clear_queue() {
        let mut queue = NeighborPriorityQueue::with_capacity(3);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        assert_eq!(queue.size(), 2);
        queue.clear();
        assert_eq!(queue.size(), 0);
        assert!(!queue.has_notvisited_node());
    }
}
