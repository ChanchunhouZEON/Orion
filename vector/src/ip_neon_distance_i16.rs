/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! NEON signed-i16 inner product for high-recall angular / MIPS search.
//!
//! Twin of [`crate::ip_neon_distance_i8`] at 16-bit precision. PA's
//! production glove100 recipe uses `-quantize_bits 16 -quantize_mode 1`
//! (signed i16 with `quant_round_inv = (max - min)/65536` scaling) on
//! the beam, with `-rerank_factor 2` running f32 IP on the top-2k
//! candidates. The 8-bit beam (`distance_ip_vector_i8`) bottoms out at
//! recall ~0.94 on glove100 because adjacent vertices' i8-quantized
//! cosines collide; 16-bit gives 256× more representable distance
//! values and recovers the high-recall band where i8 loses to PA.
//!
//! Returns **`-Σ (a_i · b_i)` as i64** so the kernel obeys the
//! "smaller == closer" contract shared with the f32 / i8 IP kernels.
//! At i16 the per-element product fits in i32 and the per-vertex sum
//! at N=100 dims stays under `100 · 32767 · 32767 ≈ 10¹¹` — within
//! i64 range with 50× headroom (i32 would overflow at high-magnitude
//! inputs, so we widen to i64 immediately via `vmlal_s16` →
//! `vpadalq_s32`).
//!
//! Throughput on Apple Silicon: 8 i16 lanes / iter × `vmull_s16`
//! (4×i32 products) × `vpadalq_s32` (widen to i64). 4-way unrolled
//! for ILP — 32 dims per outer iter, scalar tail for the last < 8.
//! Per-distance compute is roughly **2× of i8** — half the SIMD lanes
//! per register, no pairwise-byte reduction step. The trade-off vs i8
//! is precision-for-throughput; on 100-dim glove this gives ~5-15 %
//! recall lift at high-L for ~1.5× distance compute time.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Single-pair i16 inner product. Returns `-Σ a_i · b_i` (i64).
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn distance_ip_vector_i16<const N: usize>(a: &[i16; N], b: &[i16; N]) -> i64 {
    unsafe {
        let mut acc0 = vdupq_n_s64(0);
        let mut acc1 = vdupq_n_s64(0);
        let mut acc2 = vdupq_n_s64(0);
        let mut acc3 = vdupq_n_s64(0);

        let ap = a.as_ptr();
        let bp = b.as_ptr();

        // 4-way unroll: 32 i16 lanes / iter (4 × 8 lanes).
        let chunks = N / 32;
        for i in 0..chunks {
            let off = i * 32;
            let va0 = vld1q_s16(ap.add(off));
            let vb0 = vld1q_s16(bp.add(off));
            let va1 = vld1q_s16(ap.add(off + 8));
            let vb1 = vld1q_s16(bp.add(off + 8));
            let va2 = vld1q_s16(ap.add(off + 16));
            let vb2 = vld1q_s16(bp.add(off + 16));
            let va3 = vld1q_s16(ap.add(off + 24));
            let vb3 = vld1q_s16(bp.add(off + 24));

            let p0_lo = vmull_s16(vget_low_s16(va0), vget_low_s16(vb0));
            let p0_hi = vmull_high_s16(va0, vb0);
            let p1_lo = vmull_s16(vget_low_s16(va1), vget_low_s16(vb1));
            let p1_hi = vmull_high_s16(va1, vb1);
            let p2_lo = vmull_s16(vget_low_s16(va2), vget_low_s16(vb2));
            let p2_hi = vmull_high_s16(va2, vb2);
            let p3_lo = vmull_s16(vget_low_s16(va3), vget_low_s16(vb3));
            let p3_hi = vmull_high_s16(va3, vb3);

            acc0 = vpadalq_s32(acc0, p0_lo);
            acc0 = vpadalq_s32(acc0, p0_hi);
            acc1 = vpadalq_s32(acc1, p1_lo);
            acc1 = vpadalq_s32(acc1, p1_hi);
            acc2 = vpadalq_s32(acc2, p2_lo);
            acc2 = vpadalq_s32(acc2, p2_hi);
            acc3 = vpadalq_s32(acc3, p3_lo);
            acc3 = vpadalq_s32(acc3, p3_hi);
        }

        // 8-lane tail.
        let mut i = chunks * 32;
        while i + 8 <= N {
            let va = vld1q_s16(ap.add(i));
            let vb = vld1q_s16(bp.add(i));
            let p_lo = vmull_s16(vget_low_s16(va), vget_low_s16(vb));
            let p_hi = vmull_high_s16(va, vb);
            acc0 = vpadalq_s32(acc0, p_lo);
            acc0 = vpadalq_s32(acc0, p_hi);
            i += 8;
        }

        let mut total = vaddvq_s64(vaddq_s64(vaddq_s64(acc0, acc1), vaddq_s64(acc2, acc3)));

        // Scalar tail for the last < 8 elements.
        while i < N {
            total += (a[i] as i64) * (b[i] as i64);
            i += 1;
        }
        -total
    }
}

/// 4 candidates × 1 query batched i16 IP. Same interleave pattern as
/// `distance_ip_vector_i8_batch4`: 4 independent i64 accumulator
/// chains across 4 base-vector streams to expose memory-level
/// parallelism through the MSHR queue.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn distance_ip_vector_i16_batch4<const N: usize>(
    a0: &[i16; N],
    a1: &[i16; N],
    a2: &[i16; N],
    a3: &[i16; N],
    q: &[i16; N],
) -> [i64; 4] {
    unsafe {
        let mut acc0 = vdupq_n_s64(0);
        let mut acc1 = vdupq_n_s64(0);
        let mut acc2 = vdupq_n_s64(0);
        let mut acc3 = vdupq_n_s64(0);

        let p0 = a0.as_ptr();
        let p1 = a1.as_ptr();
        let p2 = a2.as_ptr();
        let p3 = a3.as_ptr();
        let pq = q.as_ptr();

        let mut i = 0usize;
        while i + 8 <= N {
            let vq = vld1q_s16(pq.add(i));
            let vq_lo = vget_low_s16(vq);

            let v0 = vld1q_s16(p0.add(i));
            let v1 = vld1q_s16(p1.add(i));
            let v2 = vld1q_s16(p2.add(i));
            let v3 = vld1q_s16(p3.add(i));

            acc0 = vpadalq_s32(acc0, vmull_s16(vget_low_s16(v0), vq_lo));
            acc0 = vpadalq_s32(acc0, vmull_high_s16(v0, vq));
            acc1 = vpadalq_s32(acc1, vmull_s16(vget_low_s16(v1), vq_lo));
            acc1 = vpadalq_s32(acc1, vmull_high_s16(v1, vq));
            acc2 = vpadalq_s32(acc2, vmull_s16(vget_low_s16(v2), vq_lo));
            acc2 = vpadalq_s32(acc2, vmull_high_s16(v2, vq));
            acc3 = vpadalq_s32(acc3, vmull_s16(vget_low_s16(v3), vq_lo));
            acc3 = vpadalq_s32(acc3, vmull_high_s16(v3, vq));

            i += 8;
        }

        let mut r = [
            vaddvq_s64(acc0),
            vaddvq_s64(acc1),
            vaddvq_s64(acc2),
            vaddvq_s64(acc3),
        ];

        while i < N {
            let qi = q[i] as i64;
            r[0] += (a0[i] as i64) * qi;
            r[1] += (a1[i] as i64) * qi;
            r[2] += (a2[i] as i64) * qi;
            r[3] += (a3[i] as i64) * qi;
            i += 1;
        }

        [-r[0], -r[1], -r[2], -r[3]]
    }
}

/// Scalar fallback.
#[cfg(not(target_arch = "aarch64"))]
#[inline(never)]
pub fn distance_ip_vector_i16<const N: usize>(a: &[i16; N], b: &[i16; N]) -> i64 {
    let mut s = 0i64;
    for i in 0..N {
        s += (a[i] as i64) * (b[i] as i64);
    }
    -s
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(never)]
pub fn distance_ip_vector_i16_batch4<const N: usize>(
    a0: &[i16; N],
    a1: &[i16; N],
    a2: &[i16; N],
    a3: &[i16; N],
    q: &[i16; N],
) -> [i64; 4] {
    [
        distance_ip_vector_i16::<N>(a0, q),
        distance_ip_vector_i16::<N>(a1, q),
        distance_ip_vector_i16::<N>(a2, q),
        distance_ip_vector_i16::<N>(a3, q),
    ]
}
