/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! NEON squared-L2 distance for u8 vectors. Used as a cheap low-precision
//! pre-filter before the full-precision f32 compare (ParlayANN-style
//! quantized filtering; see `beamSearch.h:165`).
//!
//! NEON pipeline: 16 × u8 per register via `vld1q_u8`, absolute-diff widen
//! to u16 via `vabdl_u8`, square-accumulate into u32 via `vmlal_u16`. Final
//! horizontal sum via `vaddvq_u32`. Returns raw integer sum — caller must
//! multiply by `slope²` to get the actual f32 distance if needed.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Squared-L2 on u8 vectors. Returns the raw integer sum `Σ (a_i - b_i)²`
/// as f32 (cast without scaling — the caller owns the quantization scale).
///
/// `N` must be a multiple of 16.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn distance_l2_vector_u8<const N: usize>(a: &[u8; N], b: &[u8; N]) -> f32 {
    debug_assert_eq!(N % 16, 0);
    unsafe {
        let mut acc0 = vdupq_n_u32(0);
        let mut acc1 = vdupq_n_u32(0);
        let mut acc2 = vdupq_n_u32(0);
        let mut acc3 = vdupq_n_u32(0);

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // Process 64 bytes per outer iteration (4 × 16-byte vectors).
        let chunks = N / 64;
        for i in 0..chunks {
            let off = i * 64;
            let va0 = vld1q_u8(a_ptr.add(off));
            let vb0 = vld1q_u8(b_ptr.add(off));
            let va1 = vld1q_u8(a_ptr.add(off + 16));
            let vb1 = vld1q_u8(b_ptr.add(off + 16));
            let va2 = vld1q_u8(a_ptr.add(off + 32));
            let vb2 = vld1q_u8(b_ptr.add(off + 32));
            let va3 = vld1q_u8(a_ptr.add(off + 48));
            let vb3 = vld1q_u8(b_ptr.add(off + 48));

            // Absolute difference widen to u16: |a-b| fits in u8 → safe u16.
            let d0_lo = vabdl_u8(vget_low_u8(va0), vget_low_u8(vb0));
            let d0_hi = vabdl_u8(vget_high_u8(va0), vget_high_u8(vb0));
            let d1_lo = vabdl_u8(vget_low_u8(va1), vget_low_u8(vb1));
            let d1_hi = vabdl_u8(vget_high_u8(va1), vget_high_u8(vb1));
            let d2_lo = vabdl_u8(vget_low_u8(va2), vget_low_u8(vb2));
            let d2_hi = vabdl_u8(vget_high_u8(va2), vget_high_u8(vb2));
            let d3_lo = vabdl_u8(vget_low_u8(va3), vget_low_u8(vb3));
            let d3_hi = vabdl_u8(vget_high_u8(va3), vget_high_u8(vb3));

            // Square-accumulate: u16² → u32.
            acc0 = vmlal_u16(acc0, vget_low_u16(d0_lo), vget_low_u16(d0_lo));
            acc0 = vmlal_u16(acc0, vget_high_u16(d0_lo), vget_high_u16(d0_lo));
            acc1 = vmlal_u16(acc1, vget_low_u16(d0_hi), vget_low_u16(d0_hi));
            acc1 = vmlal_u16(acc1, vget_high_u16(d0_hi), vget_high_u16(d0_hi));
            acc2 = vmlal_u16(acc2, vget_low_u16(d1_lo), vget_low_u16(d1_lo));
            acc2 = vmlal_u16(acc2, vget_high_u16(d1_lo), vget_high_u16(d1_lo));
            acc3 = vmlal_u16(acc3, vget_low_u16(d1_hi), vget_low_u16(d1_hi));
            acc3 = vmlal_u16(acc3, vget_high_u16(d1_hi), vget_high_u16(d1_hi));
            acc0 = vmlal_u16(acc0, vget_low_u16(d2_lo), vget_low_u16(d2_lo));
            acc0 = vmlal_u16(acc0, vget_high_u16(d2_lo), vget_high_u16(d2_lo));
            acc1 = vmlal_u16(acc1, vget_low_u16(d2_hi), vget_low_u16(d2_hi));
            acc1 = vmlal_u16(acc1, vget_high_u16(d2_hi), vget_high_u16(d2_hi));
            acc2 = vmlal_u16(acc2, vget_low_u16(d3_lo), vget_low_u16(d3_lo));
            acc2 = vmlal_u16(acc2, vget_high_u16(d3_lo), vget_high_u16(d3_lo));
            acc3 = vmlal_u16(acc3, vget_low_u16(d3_hi), vget_low_u16(d3_hi));
            acc3 = vmlal_u16(acc3, vget_high_u16(d3_hi), vget_high_u16(d3_hi));
        }

        // Tail: remaining 16-byte chunks (N % 64 != 0 case).
        let remaining_start = chunks * 64;
        let mut i = remaining_start;
        while i + 16 <= N {
            let va = vld1q_u8(a_ptr.add(i));
            let vb = vld1q_u8(b_ptr.add(i));
            let d_lo = vabdl_u8(vget_low_u8(va), vget_low_u8(vb));
            let d_hi = vabdl_u8(vget_high_u8(va), vget_high_u8(vb));
            acc0 = vmlal_u16(acc0, vget_low_u16(d_lo), vget_low_u16(d_lo));
            acc0 = vmlal_u16(acc0, vget_high_u16(d_lo), vget_high_u16(d_lo));
            acc1 = vmlal_u16(acc1, vget_low_u16(d_hi), vget_low_u16(d_hi));
            acc1 = vmlal_u16(acc1, vget_high_u16(d_hi), vget_high_u16(d_hi));
            i += 16;
        }

        let acc = vaddq_u32(vaddq_u32(acc0, acc1), vaddq_u32(acc2, acc3));
        vaddvq_u32(acc) as f32
    }
}

// Scalar fallback for non-aarch64.
#[cfg(not(target_arch = "aarch64"))]
#[inline(never)]
pub fn distance_l2_vector_u8<const N: usize>(a: &[u8; N], b: &[u8; N]) -> f32 {
    let mut sum: u32 = 0;
    for i in 0..N {
        let d = (a[i] as i16 - b[i] as i16).unsigned_abs() as u32;
        sum += d * d;
    }
    sum as f32
}

/// Batched u8 L2: 4 candidates × 1 query in a single pass.
///
/// Interleaves 4 independent candidate streams at each 16-byte chunk,
/// letting the OoO core overlap 4 DRAM loads and keep 4 independent
/// accumulator dependency chains running in parallel. Amortizes the
/// per-candidate dispatch overhead and improves memory-level parallelism
/// beyond what serial `distance_l2_vector_u8` can extract.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn distance_l2_vector_u8_batch4<const N: usize>(
    a0: &[u8; N],
    a1: &[u8; N],
    a2: &[u8; N],
    a3: &[u8; N],
    q: &[u8; N],
) -> [f32; 4] {
    debug_assert_eq!(N % 16, 0);
    unsafe {
        let mut acc0 = vdupq_n_u32(0);
        let mut acc1 = vdupq_n_u32(0);
        let mut acc2 = vdupq_n_u32(0);
        let mut acc3 = vdupq_n_u32(0);

        let p0 = a0.as_ptr();
        let p1 = a1.as_ptr();
        let p2 = a2.as_ptr();
        let p3 = a3.as_ptr();
        let pq = q.as_ptr();

        let mut i = 0usize;
        while i + 16 <= N {
            let vq = vld1q_u8(pq.add(i));

            let v0 = vld1q_u8(p0.add(i));
            let v1 = vld1q_u8(p1.add(i));
            let v2 = vld1q_u8(p2.add(i));
            let v3 = vld1q_u8(p3.add(i));

            // Candidate 0.
            let d_lo = vabdl_u8(vget_low_u8(v0), vget_low_u8(vq));
            let d_hi = vabdl_u8(vget_high_u8(v0), vget_high_u8(vq));
            acc0 = vmlal_u16(acc0, vget_low_u16(d_lo), vget_low_u16(d_lo));
            acc0 = vmlal_u16(acc0, vget_high_u16(d_lo), vget_high_u16(d_lo));
            acc0 = vmlal_u16(acc0, vget_low_u16(d_hi), vget_low_u16(d_hi));
            acc0 = vmlal_u16(acc0, vget_high_u16(d_hi), vget_high_u16(d_hi));

            // Candidate 1.
            let d_lo = vabdl_u8(vget_low_u8(v1), vget_low_u8(vq));
            let d_hi = vabdl_u8(vget_high_u8(v1), vget_high_u8(vq));
            acc1 = vmlal_u16(acc1, vget_low_u16(d_lo), vget_low_u16(d_lo));
            acc1 = vmlal_u16(acc1, vget_high_u16(d_lo), vget_high_u16(d_lo));
            acc1 = vmlal_u16(acc1, vget_low_u16(d_hi), vget_low_u16(d_hi));
            acc1 = vmlal_u16(acc1, vget_high_u16(d_hi), vget_high_u16(d_hi));

            // Candidate 2.
            let d_lo = vabdl_u8(vget_low_u8(v2), vget_low_u8(vq));
            let d_hi = vabdl_u8(vget_high_u8(v2), vget_high_u8(vq));
            acc2 = vmlal_u16(acc2, vget_low_u16(d_lo), vget_low_u16(d_lo));
            acc2 = vmlal_u16(acc2, vget_high_u16(d_lo), vget_high_u16(d_lo));
            acc2 = vmlal_u16(acc2, vget_low_u16(d_hi), vget_low_u16(d_hi));
            acc2 = vmlal_u16(acc2, vget_high_u16(d_hi), vget_high_u16(d_hi));

            // Candidate 3.
            let d_lo = vabdl_u8(vget_low_u8(v3), vget_low_u8(vq));
            let d_hi = vabdl_u8(vget_high_u8(v3), vget_high_u8(vq));
            acc3 = vmlal_u16(acc3, vget_low_u16(d_lo), vget_low_u16(d_lo));
            acc3 = vmlal_u16(acc3, vget_high_u16(d_lo), vget_high_u16(d_lo));
            acc3 = vmlal_u16(acc3, vget_low_u16(d_hi), vget_low_u16(d_hi));
            acc3 = vmlal_u16(acc3, vget_high_u16(d_hi), vget_high_u16(d_hi));

            i += 16;
        }

        [
            vaddvq_u32(acc0) as f32,
            vaddvq_u32(acc1) as f32,
            vaddvq_u32(acc2) as f32,
            vaddvq_u32(acc3) as f32,
        ]
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(never)]
pub fn distance_l2_vector_u8_batch4<const N: usize>(
    a0: &[u8; N],
    a1: &[u8; N],
    a2: &[u8; N],
    a3: &[u8; N],
    q: &[u8; N],
) -> [f32; 4] {
    [
        distance_l2_vector_u8(a0, q),
        distance_l2_vector_u8(a1, q),
        distance_l2_vector_u8(a2, q),
        distance_l2_vector_u8(a3, q),
    ]
}
