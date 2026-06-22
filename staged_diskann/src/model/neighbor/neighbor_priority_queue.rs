/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::model::Neighbor;

/// Round `n` up to the next multiple of 16 — sized so the underlying
/// `Vec<Neighbor>` (12-byte payload, 4-byte aligned) ends on a
/// 192-byte boundary (= 12 × 16-B NEON registers, 1.5 × 128-B M2
/// cache lines, integer multiple of the 64-B prefetch granule). All
/// allocations use `pad16(capacity + 1)`: the `+1` accounts for the
/// sentinel slot the `insert` path can write to (when `lo == capacity`
/// on a tie-break), and the outer `pad16` guarantees the *whole*
/// buffer length is a 16-multiple — so `merge_scratch` (sized
/// identically) and `pq.data` can `mem::swap` without resize and the
/// SIMD `copy_within` on the active prefix has even-count vector ops
/// across the full padded region. Doubling the alignment from 8 → 16
/// targets the residual L%8≠0 jitter (CV mean 29% vs aligned 22%
/// in the pad8 run): with pad16, *every* L's active prefix is at
/// least 16-entry-aligned in the buffer, killing the mod-8 phase.
/// The user-visible `capacity` stays at the requested value.
#[inline]
pub const fn pad16(n: usize) -> usize {
    (n + 15) & !15
}

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
            data: vec![Neighbor::default(); pad16(capacity + 1)],
        }
    }

    pub fn insert(&mut self, nbr: Neighbor) -> bool {
        if self.size == self.capacity && self.get_at(self.size - 1) < &nbr {
            return false;
        }

        let mut lo = 0;
        let mut hi = self.size;
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if &nbr < self.get_at(mid) {
                hi = mid;
            } else if self.get_at(mid).id == nbr.id {
                return false;
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

        true
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
        let needed = pad16(capacity + 1);
        if needed > self.data.len() {
            self.data.resize(needed, Neighbor::default());
        }
        self.capacity = capacity;
    }

    pub fn reserve(&mut self, capacity: usize) {
        if capacity > self.capacity {
            let needed = pad16(capacity + 1);
            if needed > self.data.len() {
                self.data.resize(needed, Neighbor::default());
            }
            self.capacity = capacity;
        }
    }

    pub fn clear(&mut self) {
        self.size = 0;
        self.cur = 0;
    }

    pub fn neighbors(&self) -> &[Neighbor] {
        &self.data[..self.size]
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
        let target_len = data_len.max(pad16(cap + 1));
        if scratch.len() < target_len {
            scratch.resize(target_len, Neighbor::default());
        }
        std::mem::swap(&mut self.data, scratch);
        self.size = new_len;
        self.cur = if new_cur == usize::MAX {
            new_len
        } else {
            new_cur
        };
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

        if self.size == self.capacity {
            let worst = &self.data[self.size - 1];
            if !sorted_cands[0].lt(worst) {
                return;
            }
        }

        let cap = self.capacity;
        let data_len = self.data.len();
        scratch.clear();
        let build_cap_hint = self.size + sorted_cands.len();
        scratch.reserve(build_cap_hint.max(data_len));

        let mut src_pos: usize = 0;
        let mut new_cur: usize = usize::MAX;

        for b in sorted_cands.iter() {
            let sub = &self.data[src_pos..self.size];
            let local_pos = sub.partition_point(|x| x.lt(b));
            let abs_pos = src_pos + local_pos;
            let dedup = abs_pos < self.size && self.data[abs_pos].id == b.id;

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

        let new_len = scratch.len().min(cap);
        let target_len = data_len.max(pad16(cap + 1));
        if scratch.len() < target_len {
            scratch.resize(target_len, Neighbor::default());
        }

        std::mem::swap(&mut self.data, scratch);
        self.size = new_len;
        self.cur = if new_cur == usize::MAX {
            new_len
        } else {
            new_cur
        };
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
    fn pad16_rounds_up() {
        assert_eq!(pad16(0), 0);
        assert_eq!(pad16(1), 16);
        assert_eq!(pad16(15), 16);
        assert_eq!(pad16(16), 16);
        assert_eq!(pad16(17), 32);
    }

    #[test]
    fn buffer_padded_to_16_multiple() {
        let pq = NeighborPriorityQueue::with_capacity(18);
        assert_eq!(pq.capacity(), 18);
        // Whole buffer is a 16-multiple.
        assert_eq!(pq.data.len() % 16, 0);
        // And it accommodates the sentinel slot at index `cap`.
        assert!(pq.data.len() > pq.capacity());
    }

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
    fn test_peek_notvisited() {
        let mut queue = NeighborPriorityQueue::with_capacity(3);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        assert_eq!(queue.peek_notvisited().map(|n| n.id), Some(2));
        queue.closest_notvisited();
        assert_eq!(queue.peek_notvisited().map(|n| n.id), Some(1));
        queue.closest_notvisited();
        assert!(queue.peek_notvisited().is_none());
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

    #[test]
    fn new_is_empty() {
        let q = NeighborPriorityQueue::new();
        assert_eq!(q.size(), 0);
        assert!(!q.has_notvisited_node());
        assert_eq!(q.neighbors().len(), 0);
    }

    #[test]
    fn default_is_empty() {
        let q = NeighborPriorityQueue::default();
        assert_eq!(q.size(), 0);
    }

    #[test]
    fn reserve_grows_buffer() {
        let mut q = NeighborPriorityQueue::with_capacity(4);
        let initial = q.data.len();
        q.reserve(100);
        assert!(q.data.len() >= initial);
    }

    #[test]
    fn set_capacity_resizes() {
        let mut q = NeighborPriorityQueue::with_capacity(8);
        q.insert(Neighbor::new(1, 0.5));
        q.insert(Neighbor::new(2, 0.6));
        q.set_capacity(4);
        assert_eq!(q.capacity(), 4);
        assert!(q.size() <= 4);
    }

    #[test]
    fn neighbors_returns_sorted_prefix() {
        let mut q = NeighborPriorityQueue::with_capacity(4);
        q.insert(Neighbor::new(10, 3.0));
        q.insert(Neighbor::new(20, 1.0));
        q.insert(Neighbor::new(30, 2.0));
        let n = q.neighbors();
        assert_eq!(n.len(), 3);
        assert!(n[0].distance <= n[1].distance);
        assert!(n[1].distance <= n[2].distance);
    }

    #[test]
    fn batch_merge_dedups_and_sorts() {
        let mut q = NeighborPriorityQueue::with_capacity(8);
        q.insert(Neighbor::new(1, 1.0));
        q.insert(Neighbor::new(2, 2.0));
        // sorted by distance asc
        let cands = [
            Neighbor::new(3, 0.5),
            Neighbor::new(4, 1.5),
            Neighbor::new(5, 3.0),
        ];
        let mut scratch = Vec::with_capacity(16);
        q.batch_merge(&cands, &mut scratch);
        let out = q.neighbors();
        assert!(out.windows(2).all(|w| w[0].distance <= w[1].distance));
        // Should include the new lower-distance candidate.
        assert!(out.iter().any(|n| n.id == 3));
    }

    #[test]
    fn batch_merge_gallop_matches_linear_merge() {
        let mut q1 = NeighborPriorityQueue::with_capacity(16);
        let mut q2 = NeighborPriorityQueue::with_capacity(16);
        for i in 0..8 {
            let n = Neighbor::new(i as u32, (i * 2) as f32);
            q1.insert(n);
            q2.insert(n);
        }
        let cands: Vec<Neighbor> = (10..18).map(|i| Neighbor::new(i, i as f32 + 0.1)).collect();
        let mut scratch1 = Vec::with_capacity(32);
        let mut scratch2 = Vec::with_capacity(32);
        q1.batch_merge(&cands, &mut scratch1);
        q2.batch_merge_gallop(&cands, &mut scratch2);
        // Both paths should produce the same ordered output.
        assert_eq!(q1.size(), q2.size());
        for i in 0..q1.size() {
            assert_eq!(q1[i].id, q2[i].id);
            assert_eq!(q1[i].distance, q2[i].distance);
        }
    }
}
