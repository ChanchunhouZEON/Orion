/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::model::Neighbor;

#[derive(Debug)]
pub struct NeighborPriorityQueue {
    size: usize,
    capacity: usize,
    cur: usize,
    data: Vec<Neighbor>,
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
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            size: 0,
            capacity,
            cur: 0,
            data: vec![Neighbor::default(); capacity + 1],
        }
    }

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
        self.data[lo] = Neighbor::new(nbr.id, nbr.distance);
        if self.size < self.capacity {
            self.size += 1;
        }
        if lo < self.cur {
            self.cur = lo;
        }
    }

    fn get_at(&self, index: usize) -> &Neighbor {
        unsafe { self.data.get_unchecked(index) }
    }

    pub fn closest_notvisited(&mut self) -> Neighbor {
        self.data[self.cur].visited = true;
        let pre = self.cur;
        while self.cur < self.size && self.get_at(self.cur).visited {
            self.cur += 1;
        }
        self.data[pre]
    }

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

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        if capacity > self.data.len() {
            self.data.resize(capacity + 1, Neighbor::default());
        }
        self.capacity = capacity;
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
    }
}

impl std::ops::Index<usize> for NeighborPriorityQueue {
    type Output = Neighbor;

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
