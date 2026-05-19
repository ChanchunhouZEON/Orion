/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! [`CacheLineDistanceBuffer`] — chunk-level partial-accumulator
//! scratch + per-vertex running fold for the deferred-consolidation
//! streaming compute path.
//!
//! ## Two-phase layout
//!
//! 1. **Compute phase** — between prefetch triggers, the caller's
//!    `K::step` calls fill UNROLL slots per round into the buffer's
//!    `scratch` (a `Box<[MaybeUninit<u8>]>` reinterpreted per-K).
//!    The slots are MaybeUninit; the caller is responsible for
//!    writing `K::init()` to each slot before `K::step` accumulates
//!    into it.
//! 2. **Merge phase** — at the lookahead boundary (same trigger
//!    that fires the next prefetch burst), the buffer folds all
//!    accumulated chunk-accs into its single per-vertex
//!    `running_acc`, emitting completed `f32` distances to `dists`.
//!    The fast path uses a `#[inline(always)]` 4-way tree-reduce
//!    helper ([`merge4_into`]); the slow path handles vertex
//!    boundaries and tail accs serially.
//!
//! ## State carried across triggers
//!
//! - `running_acc` + `chunks_done` — partial fold of the
//!   in-progress vertex (always lives in the buffer; loaded once
//!   per merge call, stored once at the end).
//! - `scratch_filled` — bytes written into `scratch` since the last
//!   merge call. Reset to 0 after merge so the next round's slots
//!   start at offset 0.
//!
//! ## Sizing
//!
//! `scratch` is sized to hold `lookahead × max_acc_size` bytes —
//! enough to buffer one full prefetch lookahead window's worth of
//! chunk accs without overflow. Grows on demand if the caller
//! reserves past the current capacity.

use crate::distance_fn::DistanceFn;
use std::mem::MaybeUninit;

/// Maximum `size_of::<K::Acc>()` across every NEON kernel we ship
/// (i16's `(int64x2_t × 4) = 64 B` is the largest).
const MAX_ACC_BYTES: usize = 64;

/// Default initial scratch capacity in **bytes**. ~4 KiB ≈ 64 i16
/// accs (64 B each) ≈ 256 i8 accs (16 B each). Covers a stride of
/// 16 lookahead lines × 4 chunks/line = 64 chunks at i16 size; grows
/// on demand otherwise.
const DEFAULT_DISTANCE_BUFFER_SIZE: usize = 4 * 1024;

/// 16-byte-aligned 64-byte scratch buffer used to hold the running
/// per-vertex `K::Acc` between merge calls without going through
/// the heap. `align(16)` satisfies every NEON kernel's
/// `align_of::<K::Acc>()`.
#[repr(C, align(16))]
#[derive(Copy, Clone)]
struct AlignedAccBuf([u8; MAX_ACC_BYTES]);

/// Tree-reduce 4 chunk accs into the running vertex acc.
/// Critical-path = 3 dependent merges (`a0+a1`, `a2+a3`, then sum,
/// then add to running) vs 4 in a serial fold. The two pair-merges
/// (a0+a1, a2+a3) are independent so they issue in parallel through
/// M2's NEON pipes; per-`K::merge` 4-way internal ILP keeps every
/// pipe busy.
///
/// `#[inline(always)]` is critical: it lets LLVM see the
/// fixed-size `&[K::Acc; 4]` reference and promote the array's
/// elements to NEON registers (no L1 round-trip), and lets the
/// caller fuse the `K::merge` chain into surrounding compute.
#[inline(always)]
fn merge4_into<K: DistanceFn>(running: &mut K::Acc, accs: &[K::Acc; 4]) {
    let mut p01 = accs[0];
    K::merge(&mut p01, accs[1]);
    let mut p23 = accs[2];
    K::merge(&mut p23, accs[3]);
    K::merge(&mut p01, p23);
    K::merge(running, p01);
}

/// Streaming reducer: chunk-acc scratch + per-vertex running fold +
/// `f32` dist output. See module docs.
pub struct CacheLineDistanceBuffer {
    /// Output buffer for the merge phase — completed per-vertex
    /// `f32` distances since the last merge call.
    dists: Vec<f32>,
    /// In-progress per-vertex running acc, stored as raw bytes for
    /// type erasure across kernels. Loaded → folded → stored once
    /// per `merge_buffered` call.
    running_acc: AlignedAccBuf,
    /// Number of chunks already folded into `running_acc` for the
    /// in-progress vertex. `0` means "no partial state".
    chunks_done: usize,
    /// Chunk-acc scratch reinterpreted per-K as `[MaybeUninit<K::Acc>]`.
    /// The caller's `K::step` writes here; `merge_buffered` reads
    /// and resets `scratch_filled` to 0.
    scratch: Box<[MaybeUninit<u8>]>,
    /// Bytes currently filled in `scratch` (since the last merge).
    /// Advances on every `slots_mut` reservation; reset to 0 at the
    /// end of each `merge_buffered` call.
    scratch_filled: usize,
}

impl Default for CacheLineDistanceBuffer {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_DISTANCE_BUFFER_SIZE)
    }
}

impl CacheLineDistanceBuffer {
    /// New buffer with a reasonable initial dists capacity (~64) and
    /// scratch capacity matching the default. Growable on demand.
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_DISTANCE_BUFFER_SIZE)
    }

    /// New buffer with `capacity` bytes of chunk-acc scratch.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            dists: Vec::with_capacity(64),
            running_acc: AlignedAccBuf([0; MAX_ACC_BYTES]),
            chunks_done: 0,
            scratch: Box::new_uninit_slice(capacity.max(DEFAULT_DISTANCE_BUFFER_SIZE)),
            scratch_filled: 0,
        }
    }

    /// Reset all in-progress state. Callers should invoke this at
    /// the start of every `DistanceStream::run`.
    #[inline]
    pub fn discard_buffer(&mut self) {
        self.dists.clear();
        self.chunks_done = 0;
        self.scratch_filled = 0;
        // `running_acc` bytes don't need clearing — `chunks_done = 0`
        // means we'll skip reading them on the next merge.
    }

    /// Ensure `extra` more bytes are writable past `scratch_filled`.
    /// Grows the backing allocation if needed (next-power-of-two).
    #[inline]
    fn ensure_room(&mut self, extra: usize) {
        let cap = self.scratch.len();
        if self.scratch_filled + extra <= cap {
            return;
        }
        let need = self.scratch_filled + extra;
        let new_cap = need.next_power_of_two().max(cap * 2);
        let mut grown: Box<[MaybeUninit<u8>]> = Box::new_uninit_slice(new_cap);
        // SAFETY: copy the live region [0..scratch_filled] from old
        // to new buffer; allocations are disjoint.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.scratch.as_ptr(),
                grown.as_mut_ptr(),
                self.scratch_filled,
            );
        }
        self.scratch = grown;
    }

    /// Reserve `n` slots in the scratch, return as
    /// `&mut [MaybeUninit<K::Acc>]`. Caller MUST initialise each
    /// slot (via `slot.write(K::init())` or equivalent) before
    /// reading or before calling `merge_buffered`.
    ///
    /// The MaybeUninit shape is deliberate: we **don't** zero-fill
    /// the slots up front. The caller's `K::step` does
    /// `acc += a*b`, which reads `acc` first — so the caller must
    /// `slot.write(K::init())` immediately before the step. LLVM
    /// can fuse the init+step into a non-accumulating "fresh write"
    /// (`vpadalq(0, x) → vpaddlq(x)` for i16, equivalent for i8),
    /// so the per-slot init is essentially free in the optimised
    /// codegen.
    #[inline]
    pub fn slots_mut<K: DistanceFn>(&mut self, n: usize) -> &mut [MaybeUninit<K::Acc>] {
        let acc_size = std::mem::size_of::<K::Acc>();
        let extra = n * acc_size;
        self.ensure_room(extra);
        let start = self.scratch_filled;
        self.scratch_filled = start + extra;
        // SAFETY: alignment guaranteed by Box::new_uninit_slice (≥16B
        // on aarch64/x86_64) and acc_size's multiple-of-16 property.
        let ptr = unsafe { self.scratch.as_mut_ptr().add(start) as *mut MaybeUninit<K::Acc> };
        debug_assert_eq!(
            (ptr as usize) % std::mem::align_of::<K::Acc>(),
            0,
            "scratch slot pointer must satisfy K::Acc alignment"
        );
        unsafe { std::slice::from_raw_parts_mut(ptr, n) }
    }

    /// Fold every chunk-acc accumulated in `scratch[0..scratch_filled]`
    /// into the running per-vertex acc, emitting completed vertex
    /// distances to `dists`. Resets `scratch_filled` to 0 so the
    /// next round writes from the start of scratch again.
    ///
    /// Fast path: 4-way tree-reduce via [`merge4_into`] when at
    /// least 4 accs are available and the next 4 don't cross a
    /// vertex boundary. Slow path handles tails and boundaries one
    /// chunk at a time.
    ///
    /// `max_verts` caps how many vertices are emitted from this
    /// merge — used by the streaming caller to defer excess
    /// vertices to a later trigger.
    #[inline]
    pub fn merge_buffered<K: DistanceFn>(
        &mut self,
        chunks_per_vert: usize,
        max_verts: usize,
    ) -> &[f32] {
        debug_assert!(chunks_per_vert > 0);
        let acc_size = std::mem::size_of::<K::Acc>();
        debug_assert!(
            acc_size <= MAX_ACC_BYTES,
            "K::Acc size {} exceeds MAX_ACC_BYTES {}",
            acc_size,
            MAX_ACC_BYTES
        );
        debug_assert_eq!(
            self.scratch_filled % acc_size,
            0,
            "scratch byte length {} not a multiple of size_of::<K::Acc>={}",
            self.scratch_filled,
            acc_size
        );
        let total = self.scratch_filled / acc_size;

        // SAFETY: caller is contracted to have initialised every
        // slot returned by `slots_mut` before this call. Alignment
        // matches K::Acc by construction.
        let ptr = self.scratch.as_ptr() as *const K::Acc;
        let accs: &[K::Acc] = unsafe { std::slice::from_raw_parts(ptr, total) };

        self.dists.clear();

        // Load running_acc into a register (or fresh K::init() if
        // no carry).
        let mut running: K::Acc = if self.chunks_done > 0 {
            // SAFETY: written by the same K's K::Acc store at the
            // end of a prior merge_buffered.
            unsafe { std::ptr::read(self.running_acc.0.as_ptr() as *const K::Acc) }
        } else {
            K::init()
        };
        let mut chunks_done = self.chunks_done;
        let mut i = 0;

        // ── Fast path: 4-way tree-reduce ──────────────────────────
        while i + 4 <= total && self.dists.len() < max_verts && chunks_done + 4 <= chunks_per_vert {
            // Const-size array reference — LLVM can promote the
            // four loads to register reads inside merge4_into.
            let chunk: &[K::Acc; 4] = unsafe { &*(accs.as_ptr().add(i) as *const [K::Acc; 4]) };
            merge4_into::<K>(&mut running, chunk);
            chunks_done += 4;
            i += 4;
            if chunks_done == chunks_per_vert {
                self.dists.push(K::reduce(running));
                running = K::init();
                chunks_done = 0;
            }
        }

        // ── Slow path: serial fold ───────────────────────────────
        while i < total && self.dists.len() < max_verts {
            K::merge(&mut running, accs[i]);
            chunks_done += 1;
            i += 1;
            if chunks_done == chunks_per_vert {
                self.dists.push(K::reduce(running));
                running = K::init();
                chunks_done = 0;
            }
        }

        // Save partial state if any.
        if chunks_done > 0 {
            unsafe {
                std::ptr::write(self.running_acc.0.as_mut_ptr() as *mut K::Acc, running);
            }
        }
        self.chunks_done = chunks_done;
        // Reset scratch for the next round window.
        self.scratch_filled = 0;

        &self.dists
    }
}
