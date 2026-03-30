use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use crossbeam::queue::ArrayQueue;
use hashbrown::HashSet;

use crate::model::Neighbor;

/// Pre-allocated scratch space for NSG search queries.
pub struct NSGScratch {
    pub visited: HashSet<u32>,
    pub candidates: BinaryHeap<Reverse<Neighbor>>,
    pub results: BinaryHeap<Neighbor>,
}

impl NSGScratch {
    pub fn new(l: usize, estimated_nodes: usize) -> Self {
        Self {
            visited: HashSet::with_capacity(estimated_nodes),
            candidates: BinaryHeap::with_capacity(l),
            results: BinaryHeap::with_capacity(l),
        }
    }

    pub fn clear(&mut self) {
        self.visited.clear();
        self.candidates.clear();
        self.results.clear();
    }
}

/// Pool of pre-allocated scratch spaces for concurrent search.
pub struct ScratchPool {
    pool: Arc<ArrayQueue<Box<NSGScratch>>>,
}

impl ScratchPool {
    pub fn new(num_threads: usize, l: usize, estimated_nodes: usize) -> Self {
        let pool = Arc::new(ArrayQueue::new(num_threads));
        for _ in 0..num_threads {
            let scratch = Box::new(NSGScratch::new(l, estimated_nodes));
            pool.push(scratch).ok();
        }
        Self { pool }
    }

    pub fn acquire(&self) -> ScratchGuard {
        loop {
            if let Some(scratch) = self.pool.pop() {
                return ScratchGuard {
                    scratch: Some(scratch),
                    pool: self.pool.clone(),
                };
            }
            std::hint::spin_loop();
        }
    }
}

/// RAII guard that returns the scratch to the pool on drop.
pub struct ScratchGuard {
    scratch: Option<Box<NSGScratch>>,
    pool: Arc<ArrayQueue<Box<NSGScratch>>>,
}

impl ScratchGuard {
    pub fn scratch(&mut self) -> &mut NSGScratch {
        self.scratch.as_deref_mut().unwrap()
    }
}

impl Drop for ScratchGuard {
    fn drop(&mut self) {
        if let Some(mut scratch) = self.scratch.take() {
            scratch.clear();
            self.pool.push(scratch).ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Neighbor;

    #[test]
    fn test_scratch_clear() {
        let mut scratch = NSGScratch::new(10, 100);
        scratch.visited.insert(1);
        scratch.candidates.push(Reverse(Neighbor::new(0, 1.0)));
        scratch.results.push(Neighbor::new(0, 1.0));

        scratch.clear();
        assert!(scratch.visited.is_empty());
        assert!(scratch.candidates.is_empty());
        assert!(scratch.results.is_empty());
    }

    #[test]
    fn test_pool_acquire_release() {
        let pool = ScratchPool::new(2, 10, 100);
        let mut g1 = pool.acquire();
        let mut g2 = pool.acquire();
        g1.scratch().visited.insert(1);
        g2.scratch().visited.insert(2);
        drop(g1);
        drop(g2);

        let g3 = pool.acquire();
        assert!(g3.scratch.as_ref().unwrap().visited.is_empty());
    }
}
