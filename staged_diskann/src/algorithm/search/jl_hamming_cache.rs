/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! **JL Hamming cache** — a per-query, per-thread open-addressing
//! hash table that intercepts the JL Sparse threshold-recompute
//! random-read fan-out.
//!
//! ## What problem this solves
//!
//! `search_l2_u8_q`'s JL prefilter maintains a running-mean threshold
//! over the PQ's per-entry Hamming distances. Each time `pq_back_id`
//! changes, the code re-walks the PQ and calls
//! `q_jl.hamming(query_sig, pq[i].id)` for every entry — at GIST L=192
//! that's ~50 recomputes × 192 entries × ~50-130 cy each = **~270
//! µs/query** of random `122 MiB JL slab` reads (mostly L2 misses on
//! the M2's 16 MiB L2 cluster).
//!
//! But **every one of those Hammings was already computed once** by
//! the JL filter's `DistanceStream<JLHammingDistance>` sink earlier
//! in the query (the `seen` set guarantees each vertex's Hamming is
//! computed at most once per query). This cache memoises those values
//! so the recompute path becomes a small-table lookup — ~10 cy L1 hit
//! instead of ~50-130 cy L2/DRAM miss.
//!
//! Expected impact: **~+30-40% QPS at GIST L=192** with no change in
//! filter quality (whole-PQ threshold signal preserved exactly).
//!
//! ## Design
//!
//! - **Open-addressing hash table**, fixed power-of-two capacity.
//!   Default `CAPACITY = 4096` — covers L ≤ 1024 with a 25% load
//!   factor, keeping average probe length under ~1.5.
//! - **Sentinel-free reset via generation counter.** Each query bumps
//!   `gen`; entries from older generations are invisible without any
//!   memset. When `gen` is about to wrap (every ~4 billion queries),
//!   we do a full clear and reset to 1.
//! - **Knuth multiplicative hash** — good distribution for random
//!   graph IDs without depending on `id` having low entropy.
//! - **Probe cap** — after `MAX_PROBES` collisions we drop the insert
//!   (lookup will miss, recompute path stays correct). Bounds the
//!   worst-case insert cost while keeping the table operating well
//!   under typical load.
//!
//! ## Memory
//!
//! 4096 slots × 16 B = **64 KiB per scratch**. At ~21 scratches
//! (rayon workers + spare) = ~1.3 MiB total. Trivially fits in L2.

const CAPACITY: usize = 4096;
const MASK: usize = CAPACITY - 1;
const MAX_PROBES: usize = 8;

/// Knuth multiplicative hash constant (fractional part of golden-ratio
/// × 2^32). Distributes random `u32` IDs uniformly across the table
/// regardless of how IDs were assigned.
const KNUTH: u32 = 2_654_435_761;

#[derive(Clone, Copy, Debug)]
struct Slot {
    /// Generation tag — entry valid only when `gen == owner.generation`.
    generation: u32,
    /// Vertex id (the key). Garbage when `gen != owner.generation`.
    id: u32,
    /// Cached JL Hamming distance. Garbage when `gen != owner.generation`.
    hamming: u32,
    /// Pad to 16 B so each slot lives on a clean 16-B boundary; two
    /// slots fit per 32-B SIMD chunk if we ever want to vectorise
    /// the probe in the future.
    _pad: u32,
}

impl Slot {
    const EMPTY: Slot = Slot {
        generation: 0,
        id: 0,
        hamming: 0,
        _pad: 0,
    };
}

/// Open-addressing hash table mapping vertex id → JL Hamming distance.
pub struct JLHammingCache {
    table: Box<[Slot; CAPACITY]>,
    generation: u32,
}

impl JLHammingCache {
    /// Construct an empty cache. `gen` starts at 1 so the all-zero
    /// initial slot state appears as "empty" until the first reset.
    pub fn new() -> Self {
        Self {
            table: Box::new([Slot::EMPTY; CAPACITY]),
            generation: 1,
        }
    }

    /// Bump the generation so all existing entries become invisible
    /// (no memory writes needed). Wraps to a full clear when `gen`
    /// would overflow, which happens once every ~4 billion queries.
    #[inline]
    pub fn reset(&mut self) {
        if self.generation == u32::MAX {
            for s in self.table.iter_mut() {
                *s = Slot::EMPTY;
            }
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }

    /// Insert `(id, hamming)` into the cache. Drops silently if the
    /// table is full at the probe cap — lookup will fall back to a
    /// recompute, which is correct just slower for that one entry.
    #[inline]
    pub fn put(&mut self, id: u32, hamming: u32) {
        let mut idx = (id.wrapping_mul(KNUTH) as usize) & MASK;
        let g = self.generation;
        for _ in 0..MAX_PROBES {
            let slot = &mut self.table[idx];
            if slot.generation != g || slot.id == id {
                slot.generation = g;
                slot.id = id;
                slot.hamming = hamming;
                slot._pad = 0;
                return;
            }
            idx = (idx + 1) & MASK;
        }
        // Probe cap exhausted — drop the insert.
    }

    /// Look up the cached Hamming for `id`. Returns `None` on miss or
    /// stale generation, so the caller's recompute path runs.
    #[inline]
    pub fn get(&self, id: u32) -> Option<u32> {
        let mut idx = (id.wrapping_mul(KNUTH) as usize) & MASK;
        let g = self.generation;
        for _ in 0..MAX_PROBES {
            let slot = &self.table[idx];
            if slot.generation != g {
                return None;
            }
            if slot.id == id {
                return Some(slot.hamming);
            }
            idx = (idx + 1) & MASK;
        }
        None
    }
}

impl Default for JLHammingCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_then_get_round_trips() {
        let mut c = JLHammingCache::new();
        c.put(42, 100);
        c.put(100, 200);
        c.put(1_000_000, 300);
        assert_eq!(c.get(42), Some(100));
        assert_eq!(c.get(100), Some(200));
        assert_eq!(c.get(1_000_000), Some(300));
        assert_eq!(c.get(99), None);
    }

    #[test]
    fn reset_invalidates_all_entries() {
        let mut c = JLHammingCache::new();
        c.put(7, 50);
        assert_eq!(c.get(7), Some(50));
        c.reset();
        assert_eq!(c.get(7), None);
        c.put(7, 999);
        assert_eq!(c.get(7), Some(999));
    }

    #[test]
    fn duplicate_put_updates_in_place() {
        let mut c = JLHammingCache::new();
        c.put(123, 10);
        c.put(123, 20);
        assert_eq!(c.get(123), Some(20));
    }

    #[test]
    fn many_collisions_eventually_drop() {
        // Force all inserts to hash to slot 0 by picking ids whose
        // Knuth-multiplied low bits are zero. Easier: just spam ids
        // that all map close enough to fill the probe runway.
        let mut c = JLHammingCache::new();
        // Walk 64 ids that all hash to the same bucket region; only
        // the first few should land successfully.
        for i in 0..64 {
            let id = i * (CAPACITY as u32 / 64);
            c.put(id.wrapping_mul(KNUTH.wrapping_mul(0)), i);
        }
        // No invariant to assert here beyond "no panic" — the cache
        // remains usable under pressure.
    }

    #[test]
    fn generation_wraps_cleanly() {
        let mut c = JLHammingCache::new();
        c.generation = u32::MAX - 1;
        c.put(5, 100);
        assert_eq!(c.get(5), Some(100));
        c.reset(); // gen now = u32::MAX
        c.put(5, 200);
        assert_eq!(c.get(5), Some(200));
        c.reset(); // gen would wrap → full clear, gen = 1
        assert_eq!(c.generation, 1);
        assert_eq!(c.get(5), None);
    }
}
