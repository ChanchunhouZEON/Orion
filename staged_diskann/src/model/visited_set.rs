/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Visited-set abstractions for in-memory beam search.
//!
//! Two implementations live behind a common [`VisitedSet`] trait:
//!
//! - [`LinearProbeSet`] — slot-level open-addressing linear probe (the
//!   ParlayANN shape). 50% load factor, table doubles in place. The
//!   long-standing default; documented behaviour matches PA's
//!   `algorithms/utils/types.h::hashset`.
//!
//! - [`BucketedSet`] — cache-line-bucketed hash table with NEON SIMD
//!   probe. Each bucket = 16 × `u32` slots = 64 B = 1 cache line.
//!   Match check loads the whole bucket in one cache-line fetch,
//!   compares against the query ID via 4× `vceqq_u32` + `vmaxvq`,
//!   then scalar-scans the bucket for the first empty slot if no
//!   match. Bucket-full triggers linear probe at the bucket level.
//!
//! [`HashsetSeen`] is the public facade that the search hot path
//! uses; it wraps either implementation behind an enum, dispatched
//! once at construction time via the `STAGED_HASHSET` env var
//! (default = `linear`, `bucketed` switches to the SIMD path). The
//! single-call dispatch is one `match` per `insert` — LLVM inlines
//! both arms and the branch is well-predicted (a single value held
//! for the lifetime of the table).

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

const SENTINEL: u32 = u32::MAX;

/// Default graph max-degree assumed when sizing the table. Matches the
/// historical hardcoded value (glove-100 PA build `-R 100`). Plumb
/// `max_degree` through the constructor when you want to right-size
/// the table for lower-R graphs (e.g. SIFT R=64, GIST R=32).
pub const DEFAULT_MAX_DEGREE: usize = 100;

/// Common API for visited-set implementations used by beam search.
/// `insert` returns `true` iff the id was newly added; the search loop
/// keys neighbour-expansion off this return so it must be exact (no
/// false negatives).
pub trait VisitedSet {
    /// Insert `id`. Returns `true` if newly added, `false` if already
    /// present. Must never falsely report new (would re-expand).
    fn insert(&mut self, id: u32) -> bool;

    /// Reset to empty without dropping the backing storage.
    fn clear(&mut self);

    /// Resize for a new search-list-size. May reallocate; may also
    /// just call [`Self::clear`] when the new target matches the
    /// existing table size.
    fn resize_for(&mut self, search_list_size: usize);

    /// Number of distinct ids currently inserted.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── Linear-probe (canonical PA shape) ───────────────────────────────────────

/// Open-addressing linear-probe set, 50% load factor, in-place double
/// on grow. Reproduces ParlayANN's `hashset`. Single-threaded.
pub struct LinearProbeSet {
    entries: Vec<u32>,
    mask: u32,
    num_entries: usize,
}

impl LinearProbeSet {
    /// `2 × (L + 1) × R` rounded up to the next power of two.
    fn target_size(search_list_size: usize, max_degree: usize) -> usize {
        let target = 2 * (search_list_size + 1) * max_degree;
        target.next_power_of_two()
    }

    pub fn new(search_list_size: usize, max_degree: usize) -> Self {
        let size = Self::target_size(search_list_size, max_degree);
        Self {
            entries: vec![SENTINEL; size],
            mask: (size as u32).wrapping_sub(1),
            num_entries: 0,
        }
    }

    #[cold]
    fn grow(&mut self) {
        let new_size = self.entries.len() * 2;
        let mut new_entries = vec![SENTINEL; new_size];
        let new_mask = (new_size as u32).wrapping_sub(1);
        let mut count = 0usize;
        for &k in &self.entries {
            if k == SENTINEL {
                continue;
            }
            let h = (k as u64).wrapping_mul(0xbf58476d1ce4e5b9);
            let mut loc = ((h >> 32) as u32 & new_mask) as usize;
            while new_entries[loc] != SENTINEL && new_entries[loc] != k {
                loc = (loc + 1) & new_mask as usize;
            }
            new_entries[loc] = k;
            count += 1;
        }
        self.entries = new_entries;
        self.mask = new_mask;
        self.num_entries = count;
    }
}

impl VisitedSet for LinearProbeSet {
    #[inline]
    fn insert(&mut self, id: u32) -> bool {
        let h = (id as u64).wrapping_mul(0xbf58476d1ce4e5b9);
        let mut loc = ((h >> 32) as u32 & self.mask) as usize;
        if unsafe { *self.entries.get_unchecked(loc) } == id {
            return false;
        }
        if self.num_entries > self.entries.len() / 2 {
            self.grow();
            loc = ((h >> 32) as u32 & self.mask) as usize;
        }
        let mask = self.mask as usize;
        if unsafe { *self.entries.get_unchecked(loc) } != SENTINEL {
            loc = (loc + 1) & mask;
            loop {
                let cur = unsafe { *self.entries.get_unchecked(loc) };
                if cur == SENTINEL || cur == id {
                    break;
                }
                loc = (loc + 1) & mask;
            }
            if unsafe { *self.entries.get_unchecked(loc) } == id {
                return false;
            }
        }
        unsafe {
            *self.entries.get_unchecked_mut(loc) = id;
        }
        self.num_entries += 1;
        true
    }

    fn clear(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.entries.as_mut_ptr(), 0xFF, self.entries.len());
        }
        self.num_entries = 0;
    }

    fn resize_for(&mut self, search_list_size: usize) {
        let size = Self::target_size(search_list_size, DEFAULT_MAX_DEGREE);
        if size != self.entries.len() {
            self.entries = vec![SENTINEL; size];
            self.mask = (size as u32).wrapping_sub(1);
            self.num_entries = 0;
        } else {
            self.clear();
        }
    }

    fn len(&self) -> usize {
        self.num_entries
    }
}

// ── Cache-line bucketed (NEON SIMD probe) ───────────────────────────────────
//
// v4 (structural parity with linear-probe): same `Vec<u32>` layout,
// same SENTINEL semantics, same hash, same `clear()`. The only
// difference is the probe granularity — bucketed walks the table in
// 16-slot blocks via SIMD, where linear walks slot-by-slot.
//
// ```text
//   total slots N = next_pow2(2(L+1)R)   ← identical to LinearProbeSet
//   bucket b covers slots [b*16 .. b*16+16)   one 64 B cache line each
//   bucket index = (linear_home_slot) >> 4 = (hash >> 4) & (num_buckets-1)
// ```
//
// No epoch, no per-bucket count, no header bytes. Insert finds the
// first SENTINEL slot in the home bucket via a register-resident scan
// after the SIMD id-match check; on overflow, linear-probes to the
// next bucket. `clear()` is the same `write_bytes` memset as linear.

/// 4 × u32 = 16 B = one NEON register, four buckets per cache line.
///
/// Initially tried 16 (= full cache line per bucket) but the SIMD
/// shape there does 4× vld + 4× vceqq + 3× vorrq + vmaxvq = ~10 NEON
/// ops per probe step. At 50% load the SENTINEL-scan averages 8+ slots
/// per insert, so most cycles are bookkeeping rather than productive
/// probing.
///
/// With 4-slot buckets the per-probe shape collapses to **1× vld +
/// 1× vceqq + vmaxvq = ~3 NEON ops**. Overflow rate is higher (P(full)
/// ≈ Poisson(2)≥4 ≈ 14% vs 0.5% for B=16), but each overflow probe
/// step is much cheaper, and at B=4 four consecutive buckets fit in
/// one cache line — so a 2-3 step probe chain still costs one cache-
/// line load.
const BUCKET_SLOTS: usize = 4;
/// `log₂(BUCKET_SLOTS)`. The bucketed home-bucket index is
/// `(hash >> BUCKET_SLOTS_LOG2) & bucket_mask`, i.e. the bucket
/// containing the slot that `LinearProbeSet` would land on for the
/// same id. Both impls therefore probe the same memory region.
const BUCKET_SLOTS_LOG2: u32 = 2;

/// Cache-line bucketed open-addressing set with NEON SIMD probe.
///
/// Memory layout is byte-for-byte identical to [`LinearProbeSet`]:
/// flat `Vec<u32>`, SENTINEL = `u32::MAX` for empty, `2 × (L+1) × R`
/// total slots rounded up to the next power of two. Both impls touch
/// the **same cache line** for a given id; the difference is purely
/// the probe shape — bucketed checks 16 slots in one SIMD compare,
/// linear walks them slot-by-slot on collisions.
pub struct BucketedSet {
    /// Flat slot storage, SENTINEL-initialised. Each 16 consecutive
    /// u32s form one cache-line-aligned bucket.
    entries: Vec<u32>,
    /// `num_buckets - 1`, always a power of two. Total slots =
    /// `(bucket_mask + 1) × BUCKET_SLOTS`, matches `LinearProbeSet`'s
    /// total slot count for the same `(L, R)`.
    bucket_mask: u32,
    num_entries: usize,
}

impl BucketedSet {
    /// Same total-slot target as `LinearProbeSet`. Number of buckets =
    /// total_slots / BUCKET_SLOTS (always pow-2 since both factors are).
    fn target_buckets(search_list_size: usize, max_degree: usize) -> usize {
        let target_slots = 2 * (search_list_size + 1) * max_degree;
        let total_slots = target_slots.next_power_of_two();
        (total_slots / BUCKET_SLOTS).max(1)
    }

    pub fn new(search_list_size: usize, max_degree: usize) -> Self {
        let nb = Self::target_buckets(search_list_size, max_degree);
        Self {
            entries: vec![SENTINEL; nb * BUCKET_SLOTS],
            bucket_mask: (nb as u32).wrapping_sub(1),
            num_entries: 0,
        }
    }

    /// Same 64-bit splittable hash as `LinearProbeSet`.
    #[inline(always)]
    fn hash(id: u32) -> u32 {
        let h = (id as u64).wrapping_mul(0xbf58476d1ce4e5b9);
        (h >> 32) as u32
    }

    /// Map an id to its home bucket. Bucket index is the cache-line
    /// index of the slot that `LinearProbeSet` would land on for the
    /// same id: `(hash >> 4) & bucket_mask` ≡ `linear_home_slot >> 4`.
    #[inline(always)]
    fn home_bucket(&self, id: u32) -> usize {
        ((Self::hash(id) >> BUCKET_SLOTS_LOG2) & self.bucket_mask) as usize
    }

    /// Rehash all live ids into a 2× table on grow. Same shape as
    /// `LinearProbeSet::grow` but probing at bucket granularity.
    #[cold]
    fn grow(&mut self) {
        let new_buckets = (self.bucket_mask as usize + 1) * 2;
        let new_mask = (new_buckets as u32).wrapping_sub(1);
        let mut new_entries = vec![SENTINEL; new_buckets * BUCKET_SLOTS];
        let mut count = 0usize;
        for &id in &self.entries {
            if id == SENTINEL {
                continue;
            }
            let mut bucket = ((Self::hash(id) >> BUCKET_SLOTS_LOG2) & new_mask) as usize;
            'outer: loop {
                let base = bucket * BUCKET_SLOTS;
                for j in 0..BUCKET_SLOTS {
                    let slot = unsafe { new_entries.get_unchecked_mut(base + j) };
                    if *slot == SENTINEL {
                        *slot = id;
                        count += 1;
                        break 'outer;
                    }
                }
                bucket = (bucket + 1) & new_mask as usize;
            }
        }
        self.entries = new_entries;
        self.bucket_mask = new_mask;
        self.num_entries = count;
    }
}

impl VisitedSet for BucketedSet {
    #[inline]
    fn insert(&mut self, id: u32) -> bool {
        // Same 50% load factor as `LinearProbeSet`.
        if self.num_entries > self.entries.len() / 2 {
            self.grow();
        }
        let mut bucket = self.home_bucket(id);
        loop {
            unsafe {
                let base_ptr = self.entries.as_ptr().add(bucket * BUCKET_SLOTS);

                // ── SIMD match check: 4-slot bucket = one NEON
                // register. Single `vld1q_u32` loads all 4 ids;
                // `vceqq_u32` lanes them against the query; `vmaxvq`
                // reduces to a single u32 (non-zero iff any lane
                // matched). Empty slots hold SENTINEL, which never
                // matches valid graph ids (id < u32::MAX).
                #[cfg(target_arch = "aarch64")]
                {
                    let id_v = vdupq_n_u32(id);
                    let s = vld1q_u32(base_ptr);
                    if vmaxvq_u32(vceqq_u32(s, id_v)) != 0 {
                        return false;
                    }
                }
                // AVX-512 path: `_mm_set1_epi32` broadcasts id into a
                // 128-bit register, `_mm_loadu_si128` reads the 4 u32
                // slots, `_mm_cmpeq_epi32` lanes them, `_mm_movemask_epi8`
                // reduces to a single mask (non-zero iff any lane
                // matched). Same logical shape as the NEON path —
                // 128-bit is enough for a 4-slot bucket.
                #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
                {
                    use std::arch::x86_64::*;
                    let id_v = _mm_set1_epi32(id as i32);
                    let s = _mm_loadu_si128(base_ptr as *const __m128i);
                    if _mm_movemask_epi8(_mm_cmpeq_epi32(s, id_v)) != 0 {
                        return false;
                    }
                }
                #[cfg(not(any(
                    target_arch = "aarch64",
                    all(target_arch = "x86_64", target_feature = "avx512f")
                )))]
                {
                    for j in 0..BUCKET_SLOTS {
                        if *base_ptr.add(j) == id {
                            return false;
                        }
                    }
                }

                // ── Find first SENTINEL via scalar scan over 4
                // register-resident slots. ~2 cycles average at 50%
                // bucket fill; SIMD position-find via vshrn movemask
                // would need more ops at this width.
                let bucket_slice = std::slice::from_raw_parts(base_ptr, BUCKET_SLOTS);
                for j in 0..BUCKET_SLOTS {
                    if *bucket_slice.get_unchecked(j) == SENTINEL {
                        *(base_ptr as *mut u32).add(j) = id;
                        self.num_entries += 1;
                        return true;
                    }
                }
            }
            // Bucket full → linear-probe to next bucket. Rare at 50%
            // load (≈ 1% of buckets are full at steady state).
            bucket = (bucket + 1) & self.bucket_mask as usize;
        }
    }

    fn clear(&mut self) {
        // Same as `LinearProbeSet`: memset the whole table to SENTINEL.
        // O(N) but compiles to a fast L1-bandwidth memset; at 128 KB
        // table size + L1 fit, ~1 µs per call.
        unsafe {
            std::ptr::write_bytes(self.entries.as_mut_ptr(), 0xFF, self.entries.len());
        }
        self.num_entries = 0;
    }

    fn resize_for(&mut self, search_list_size: usize) {
        let nb = Self::target_buckets(search_list_size, DEFAULT_MAX_DEGREE);
        let total = nb * BUCKET_SLOTS;
        if total != self.entries.len() {
            self.entries = vec![SENTINEL; total];
            self.bucket_mask = (nb as u32).wrapping_sub(1);
            self.num_entries = 0;
        } else {
            self.clear();
        }
    }

    fn len(&self) -> usize {
        self.num_entries
    }
}

// ── Public facade — preserves the historical `HashsetSeen` API ──────────────

/// Implementation pick at construction time. Sticks for the lifetime
/// of the table — `STAGED_HASHSET` env var read once in `new()`.
enum HashsetInner {
    Linear(LinearProbeSet),
    Bucketed(BucketedSet),
}

/// L1-resident approximate hash-set visited tracker (ParlayANN-style).
///
/// Public API stays the same; the implementation is selected at
/// construction time from the `STAGED_HASHSET` env var:
///
/// - `linear` (default) — [`LinearProbeSet`], the canonical PA shape.
/// - `bucketed`         — [`BucketedSet`], NEON cache-line probe.
///
/// Run with `STAGED_HASHSET=bucketed cargo run …` to A/B against
/// the linear-probe baseline. The dedicated profile binary
/// `benchmark/src/bin/hashset_profile.rs` exercises both impls
/// directly without going through the env-var dispatch.
pub struct HashsetSeen {
    inner: HashsetInner,
}

impl HashsetSeen {
    pub fn new(search_list_size: usize) -> Self {
        let bucketed = std::env::var("STAGED_HASHSET")
            .map(|s| s.eq_ignore_ascii_case("bucketed"))
            .unwrap_or(false);
        let inner = if bucketed {
            HashsetInner::Bucketed(BucketedSet::new(search_list_size, DEFAULT_MAX_DEGREE))
        } else {
            HashsetInner::Linear(LinearProbeSet::new(search_list_size, DEFAULT_MAX_DEGREE))
        };
        Self { inner }
    }

    #[inline]
    pub fn insert(&mut self, id: u32) -> bool {
        match &mut self.inner {
            HashsetInner::Linear(s) => s.insert(id),
            HashsetInner::Bucketed(s) => s.insert(id),
        }
    }

    pub fn clear(&mut self) {
        match &mut self.inner {
            HashsetInner::Linear(s) => s.clear(),
            HashsetInner::Bucketed(s) => s.clear(),
        }
    }

    pub fn resize_for(&mut self, search_list_size: usize) {
        match &mut self.inner {
            HashsetInner::Linear(s) => s.resize_for(search_list_size),
            HashsetInner::Bucketed(s) => s.resize_for(search_list_size),
        }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        match &self.inner {
            HashsetInner::Linear(s) => s.len(),
            HashsetInner::Bucketed(s) => s.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exercise_impl<S: VisitedSet>(set: &mut S) {
        // Inserts return true the first time, false thereafter.
        assert!(set.insert(42));
        assert!(!set.insert(42));
        assert!(set.insert(100));
        assert!(set.insert(0));
        assert!(set.insert(SENTINEL.wrapping_sub(1)));
        assert!(!set.insert(100));
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn linear_basic() {
        exercise_impl(&mut LinearProbeSet::new(64, 64));
    }

    #[test]
    fn bucketed_basic() {
        exercise_impl(&mut BucketedSet::new(64, 64));
    }

    #[test]
    fn impls_agree_on_random_workload() {
        // Same id stream into both impls — every insert decision must match.
        let mut linear = LinearProbeSet::new(48, 64);
        let mut bucketed = BucketedSet::new(48, 64);
        let mut seed = 0x9E3779B97F4A7C15u64;
        for _ in 0..50_000 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let id = (seed >> 32) as u32 & 0x000F_FFFF; // 1M id space
            assert_eq!(
                linear.insert(id),
                bucketed.insert(id),
                "divergence at id={id}"
            );
        }
        assert_eq!(linear.len(), bucketed.len());
    }

    #[test]
    fn clear_resets() {
        let mut s = BucketedSet::new(16, 64);
        for i in 0..100 {
            s.insert(i);
        }
        s.clear();
        assert_eq!(s.len(), 0);
        // Re-inserting all the same ids should report new each time.
        for i in 0..100 {
            assert!(s.insert(i));
        }
    }

    #[test]
    fn linear_resize_for_grows() {
        let mut s = LinearProbeSet::new(16, 64);
        s.insert(1);
        s.insert(2);
        s.resize_for(256);
        // After resize, the previous entries are cleared.
        assert_eq!(s.len(), 0);
        // And it accepts new entries cleanly.
        assert!(s.insert(1));
        assert!(s.insert(2));
    }

    #[test]
    fn bucketed_resize_for_grows() {
        let mut s = BucketedSet::new(16, 64);
        s.insert(1);
        s.insert(2);
        s.resize_for(256);
        assert_eq!(s.len(), 0);
        assert!(s.insert(1));
        assert!(s.insert(2));
    }

    #[test]
    fn hashset_seen_default_path_basic() {
        // Default path (no STAGED_HASHSET env) — linear probe.
        let mut s = HashsetSeen::new(32);
        assert!(s.insert(7));
        assert!(!s.insert(7));
        assert!(s.insert(11));
        assert_eq!(s.len(), 2);
        s.clear();
        assert_eq!(s.len(), 0);
        s.resize_for(64);
    }
}
