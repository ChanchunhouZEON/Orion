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

    /// ParlayANN-style linear set-union merge with `mem::swap` ending.
    /// `sorted_cands` must be sorted by distance ascending. Streams both
    /// inputs in one linear pass (cache-friendly at typical L), installs
    /// the result via `mem::swap` instead of a copy-back memcpy.
    ///
    /// Preserves `visited` bits on dedup'd existing entries. Best when
    /// `sorted_cands.len()` is comparable to `self.size` — for large L
    /// or admits > 24 per hop.
    pub fn batch_merge(&mut self, sorted_cands: &[Neighbor], scratch: &mut Vec<Neighbor>) {
        if sorted_cands.is_empty() {
            return;
        }

        if self.size == self.capacity {
            let worst = &self.data[self.size - 1];
            if !sorted_cands[0].lt(worst) {
                return;
            }
        }

        let cap = self.capacity;
        let data_len = self.data.len();
        scratch.clear();
        scratch.reserve(data_len);

        let mut i = 0usize;
        let mut j = 0usize;
        let mut new_cur: usize = usize::MAX;

        while i < self.size && j < sorted_cands.len() && scratch.len() < cap {
            let a = unsafe { *self.data.get_unchecked(i) };
            let b = unsafe { *sorted_cands.get_unchecked(j) };
            if a.id == b.id {
                if !a.visited && new_cur == usize::MAX {
                    new_cur = scratch.len();
                }
                scratch.push(a);
                i += 1;
                j += 1;
            } else if a.lt(&b) {
                if !a.visited && new_cur == usize::MAX {
                    new_cur = scratch.len();
                }
                scratch.push(a);
                i += 1;
            } else {
                if !b.visited && new_cur == usize::MAX {
                    new_cur = scratch.len();
                }
                scratch.push(b);
                j += 1;
            }
        }
        while i < self.size && scratch.len() < cap {
            let a = unsafe { *self.data.get_unchecked(i) };
            if !a.visited && new_cur == usize::MAX {
                new_cur = scratch.len();
            }
            scratch.push(a);
            i += 1;
        }
        while j < sorted_cands.len() && scratch.len() < cap {
            let b = unsafe { *sorted_cands.get_unchecked(j) };
            if !b.visited && new_cur == usize::MAX {
                new_cur = scratch.len();
            }
            scratch.push(b);
            j += 1;
        }

        let new_len = scratch.len();
        let target_len = data_len.max(cap + 1);
        if scratch.len() < target_len {
            scratch.resize(target_len, Neighbor::default());
        }
        std::mem::swap(&mut self.data, scratch);
        self.size = new_len;
        self.cur = if new_cur == usize::MAX { new_len } else { new_cur };
    }

    /// Galloping binary-search merge for the case where `sorted_cands`
    /// accumulates across many hops with sparse admits (converged-phase
    /// cross-hop batching). For each candidate, `partition_point` on the
    /// shrinking suffix `self.data[src_pos..size]` gives a log-cost
    /// insertion point; runs of existing entries between insertion points
    /// are bulk-copied in one `extend_from_slice`. Final install via
    /// `mem::swap`.
    ///
    /// Preserves `visited` bits; scratch may briefly exceed `cap` during
    /// build — logical `self.size` is truncated at swap time.
    pub fn batch_merge_gallop(&mut self, sorted_cands: &[Neighbor], scratch: &mut Vec<Neighbor>) {
        if sorted_cands.is_empty() {
            return;
        }

        // Fast path: PQ full and best cand not better than PQ's worst.
        if self.size == self.capacity {
            let worst = &self.data[self.size - 1];
            if !sorted_cands[0].lt(worst) {
                return;
            }
        }

        let cap = self.capacity;
        let data_len = self.data.len(); // ≥ cap + 1 by pq invariant
        scratch.clear();
        // Overallocate: scratch may briefly exceed cap during build (we
        // truncate the logical size at the end). Upper bound is self.size +
        // sorted_cands.len().
        let build_cap_hint = self.size + sorted_cands.len();
        scratch.reserve(build_cap_hint.max(data_len));

        let mut src_pos: usize = 0;
        let mut new_cur: usize = usize::MAX;

        for b in sorted_cands.iter() {
            // Binary search for insertion point of b in self.data[src_pos..size].
            // Shrinking suffix gives gallop-like amortized cost.
            let sub = &self.data[src_pos..self.size];
            let local_pos = sub.partition_point(|x| x.lt(b));
            let abs_pos = src_pos + local_pos;
            let dedup = abs_pos < self.size && self.data[abs_pos].id == b.id;

            // Bulk-copy self.data[src_pos..abs_pos] into scratch.
            if abs_pos > src_pos {
                let run = &self.data[src_pos..abs_pos];
                if new_cur == usize::MAX {
                    for (i, n) in run.iter().enumerate() {
                        if !n.visited {
                            new_cur = scratch.len() + i;
                            break;
                        }
                    }
                }
                scratch.extend_from_slice(run);
            }

            // Emit b (or its dedup'd existing twin, preserving visited).
            if dedup {
                let existing = self.data[abs_pos];
                if !existing.visited && new_cur == usize::MAX {
                    new_cur = scratch.len();
                }
                scratch.push(existing);
                src_pos = abs_pos + 1;
            } else {
                if !b.visited && new_cur == usize::MAX {
                    new_cur = scratch.len();
                }
                scratch.push(*b);
                src_pos = abs_pos;
            }
        }

        // Tail: remaining self.data[src_pos..size] in one bulk copy.
        if src_pos < self.size {
            let run = &self.data[src_pos..self.size];
            if new_cur == usize::MAX {
                for (i, x) in run.iter().enumerate() {
                    if !x.visited {
                        new_cur = scratch.len() + i;
                        break;
                    }
                }
            }
            scratch.extend_from_slice(run);
        }

        // Logical size capped at `capacity`; entries beyond are ignored.
        let new_len = scratch.len().min(cap);
        // Ensure post-swap self.data.len() ≥ max(data_len, new_len + 1) so
        // the pq invariant (data.len() ≥ capacity + 1) and future insert()
        // copy_within(lo..size, lo+1) remain safe.
        let target_len = data_len.max(cap + 1);
        if scratch.len() < target_len {
            scratch.resize(target_len, Neighbor::default());
        }

        std::mem::swap(&mut self.data, scratch);
        self.size = new_len;
        self.cur = if new_cur == usize::MAX { new_len } else { new_cur };
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
