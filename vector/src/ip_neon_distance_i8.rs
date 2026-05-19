/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! NEON signed-i8 inner product for angular / MIPS search on unit-
//! normalized data.
//!
//! This is the kernel PA uses end-to-end for GloVe / DeepVis / t2i-1:
//! all beam-search distances run at 8-bit integer precision; only the
//! final top-`k × rerank_factor` candidates are recomputed in higher
//! precision. Their scalar loop (`mips_point.h::distance_8`) is what
//! Clang auto-vectorizes to `vmull_s8 + vpadalq_s16`, which is exactly
//! what this file materializes explicitly for repeatability.
//!
//! Returns **`-Σ (a_i · b_i)` as i32** so the kernel obeys the
//! "smaller == closer" contract shared with the f32 IP / L2 kernels.
//! Callers that need an f32 distance scale by `1 / (S × S)` where
//! `S = 127` is the i8 quantization scale.
//!
//! Throughput on Apple Silicon: 16 i8 lanes / iter × `vmull_s8` (8×i16
//! products) × `vpadalq_s16` (widen+accumulate to i32). 4-way unrolled
//! for ILP — 64 dims per outer iter, scalar tail for the last < 16.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Single-pair i8 inner product. Returns `-Σ a_i · b_i` (i32).
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn distance_ip_vector_i8<const N: usize>(a: &[i8; N], b: &[i8; N]) -> i32 {
    unsafe {
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut acc2 = vdupq_n_s32(0);
        let mut acc3 = vdupq_n_s32(0);

        let ap = a.as_ptr();
        let bp = b.as_ptr();

        // 4-way unroll: 64 i8 lanes / iter.
        let chunks = N / 64;
        for i in 0..chunks {
            let off = i * 64;
            let va0 = vld1q_s8(ap.add(off));
            let vb0 = vld1q_s8(bp.add(off));
            let va1 = vld1q_s8(ap.add(off + 16));
            let vb1 = vld1q_s8(bp.add(off + 16));
            let va2 = vld1q_s8(ap.add(off + 32));
            let vb2 = vld1q_s8(bp.add(off + 32));
            let va3 = vld1q_s8(ap.add(off + 48));
            let vb3 = vld1q_s8(bp.add(off + 48));

            let p0_lo = vmull_s8(vget_low_s8(va0), vget_low_s8(vb0));
            let p0_hi = vmull_high_s8(va0, vb0);
            let p1_lo = vmull_s8(vget_low_s8(va1), vget_low_s8(vb1));
            let p1_hi = vmull_high_s8(va1, vb1);
            let p2_lo = vmull_s8(vget_low_s8(va2), vget_low_s8(vb2));
            let p2_hi = vmull_high_s8(va2, vb2);
            let p3_lo = vmull_s8(vget_low_s8(va3), vget_low_s8(vb3));
            let p3_hi = vmull_high_s8(va3, vb3);

            acc0 = vpadalq_s16(acc0, p0_lo);
            acc0 = vpadalq_s16(acc0, p0_hi);
            acc1 = vpadalq_s16(acc1, p1_lo);
            acc1 = vpadalq_s16(acc1, p1_hi);
            acc2 = vpadalq_s16(acc2, p2_lo);
            acc2 = vpadalq_s16(acc2, p2_hi);
            acc3 = vpadalq_s16(acc3, p3_lo);
            acc3 = vpadalq_s16(acc3, p3_hi);
        }

        // 16-lane tail.
        let mut i = chunks * 64;
        while i + 16 <= N {
            let va = vld1q_s8(ap.add(i));
            let vb = vld1q_s8(bp.add(i));
            let p_lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
            let p_hi = vmull_high_s8(va, vb);
            acc0 = vpadalq_s16(acc0, p_lo);
            acc0 = vpadalq_s16(acc0, p_hi);
            i += 16;
        }

        let mut total = vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3)));

        // Scalar tail for the last < 16 elements.
        while i < N {
            total += (a[i] as i32) * (b[i] as i32);
            i += 1;
        }
        -total
    }
}

/// 4 candidates × 1 query batched i8 IP. Interleaves 4 independent
/// accumulator chains across 4 base-vector streams for memory-level
/// parallelism, mirroring the `distance_l2_vector_u8_batch4` pattern.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn distance_ip_vector_i8_batch4<const N: usize>(
    a0: &[i8; N],
    a1: &[i8; N],
    a2: &[i8; N],
    a3: &[i8; N],
    q: &[i8; N],
) -> [i32; 4] {
    unsafe {
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut acc2 = vdupq_n_s32(0);
        let mut acc3 = vdupq_n_s32(0);

        let p0 = a0.as_ptr();
        let p1 = a1.as_ptr();
        let p2 = a2.as_ptr();
        let p3 = a3.as_ptr();
        let pq = q.as_ptr();

        let mut i = 0usize;
        while i + 16 <= N {
            let vq = vld1q_s8(pq.add(i));
            let vq_lo = vget_low_s8(vq);

            let v0 = vld1q_s8(p0.add(i));
            let v1 = vld1q_s8(p1.add(i));
            let v2 = vld1q_s8(p2.add(i));
            let v3 = vld1q_s8(p3.add(i));

            acc0 = vpadalq_s16(acc0, vmull_s8(vget_low_s8(v0), vq_lo));
            acc0 = vpadalq_s16(acc0, vmull_high_s8(v0, vq));
            acc1 = vpadalq_s16(acc1, vmull_s8(vget_low_s8(v1), vq_lo));
            acc1 = vpadalq_s16(acc1, vmull_high_s8(v1, vq));
            acc2 = vpadalq_s16(acc2, vmull_s8(vget_low_s8(v2), vq_lo));
            acc2 = vpadalq_s16(acc2, vmull_high_s8(v2, vq));
            acc3 = vpadalq_s16(acc3, vmull_s8(vget_low_s8(v3), vq_lo));
            acc3 = vpadalq_s16(acc3, vmull_high_s8(v3, vq));

            i += 16;
        }

        let mut r = [
            vaddvq_s32(acc0),
            vaddvq_s32(acc1),
            vaddvq_s32(acc2),
            vaddvq_s32(acc3),
        ];

        while i < N {
            let qi = q[i] as i32;
            r[0] += (a0[i] as i32) * qi;
            r[1] += (a1[i] as i32) * qi;
            r[2] += (a2[i] as i32) * qi;
            r[3] += (a3[i] as i32) * qi;
            i += 1;
        }

        [-r[0], -r[1], -r[2], -r[3]]
    }
}

/// Scalar fallback.
#[cfg(not(target_arch = "aarch64"))]
#[inline(never)]
pub fn distance_ip_vector_i8<const N: usize>(a: &[i8; N], b: &[i8; N]) -> i32 {
    let mut s = 0i32;
    for i in 0..N {
        s += (a[i] as i32) * (b[i] as i32);
    }
    -s
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(never)]
pub fn distance_ip_vector_i8_batch4<const N: usize>(
    a0: &[i8; N],
    a1: &[i8; N],
    a2: &[i8; N],
    a3: &[i8; N],
    q: &[i8; N],
) -> [i32; 4] {
    [
        distance_ip_vector_i8::<N>(a0, q),
        distance_ip_vector_i8::<N>(a1, q),
        distance_ip_vector_i8::<N>(a2, q),
        distance_ip_vector_i8::<N>(a3, q),
    ]
}
