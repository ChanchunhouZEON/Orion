/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! NEON negative-inner-product distance for aarch64.
//!
//! On **unit-normalized** vectors, ranking by `-⟨a,b⟩` is identical to
//! ranking by squared-L2 (since `‖a−b‖² = 2 − 2⟨a,b⟩`), so this kernel
//! is a drop-in replacement for `distance_l2_vector_f32` whenever the
//! caller has pre-normalized both base and query.
//!
//! Cost advantage over L2: one FMA per element instead of sub+FMA —
//! roughly **half** the instructions on the hot inner loop.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Returns `-⟨a, b⟩` so that **smaller == closer**, matching the L2²
/// contract used by the rest of the search code.
///
/// 4-way unrolled (16 floats per iter). Requires 16-byte alignment of
/// both inputs (guaranteed by `AlignedBoxWithSlice` for the dataset and
/// `AlignedQuery` for the query buffer).
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn distance_ip_vector_f32<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    debug_assert_eq!(N % 4, 0);
    unsafe {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut s2 = vdupq_n_f32(0.0);
        let mut s3 = vdupq_n_f32(0.0);

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        const PF_AHEAD: usize = 4;
        let chunks = N / 16;
        for i in 0..chunks {
            let offset = i * 16;

            if i + PF_AHEAD < chunks {
                let pf_offset = (i + PF_AHEAD) * 16;
                let pa = a_ptr.add(pf_offset) as *const u8;
                let pb = b_ptr.add(pf_offset) as *const u8;
                std::arch::asm!(
                    "prfm pldl1keep, [{a}]",
                    "prfm pldl1keep, [{b}]",
                    a = in(reg) pa,
                    b = in(reg) pb,
                    options(nostack, preserves_flags),
                );
            }

            let a0 = vld1q_f32(a_ptr.add(offset));
            let b0 = vld1q_f32(b_ptr.add(offset));
            s0 = vfmaq_f32(s0, a0, b0);

            let a1 = vld1q_f32(a_ptr.add(offset + 4));
            let b1 = vld1q_f32(b_ptr.add(offset + 4));
            s1 = vfmaq_f32(s1, a1, b1);

            let a2 = vld1q_f32(a_ptr.add(offset + 8));
            let b2 = vld1q_f32(b_ptr.add(offset + 8));
            s2 = vfmaq_f32(s2, a2, b2);

            let a3 = vld1q_f32(a_ptr.add(offset + 12));
            let b3 = vld1q_f32(b_ptr.add(offset + 12));
            s3 = vfmaq_f32(s3, a3, b3);
        }

        let remaining_start = chunks * 16;
        for i in (remaining_start..N).step_by(4) {
            let av = vld1q_f32(a_ptr.add(i));
            let bv = vld1q_f32(b_ptr.add(i));
            s0 = vfmaq_f32(s0, av, bv);
        }

        let total = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
        -vaddvq_f32(total)
    }
}

/// Scalar fallback for non-aarch64 targets.
#[cfg(not(target_arch = "aarch64"))]
#[inline(never)]
pub fn distance_ip_vector_f32<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..N {
        s += a[i] * b[i];
    }
    -s
}

/// 4 candidates × 1 query batched f32 IP — returns `-Σ(a_k · q)` for
/// `k ∈ {0..4}` (same "smaller == closer" contract as the single-point
/// [`distance_ip_vector_f32`]).
///
/// This kernel is intended for **cold-cache** paths such as the MIPS-Q
/// final-rerank stage: the beam search ran entirely against the i8
/// quantized dataset, so `dataset` (f32) is cold when rerank starts.
/// Serial single-point calls serialize on DRAM misses; the batch-4
/// variant interleaves 4 independent accumulator chains so 4 concurrent
/// misses can overlap through the MSHR queue — giving better memory-
/// level parallelism without touching the existing hot single-point
/// kernel, which is already ILP-saturated by its 4-way internal unroll
/// and would **regress** 5-13% if we batched the whole hop loop instead.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn distance_ip_vector_f32_batch4<const N: usize>(
    a0: &[f32; N],
    a1: &[f32; N],
    a2: &[f32; N],
    a3: &[f32; N],
    q: &[f32; N],
) -> [f32; 4] {
    debug_assert_eq!(N % 4, 0);
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let p0 = a0.as_ptr();
        let p1 = a1.as_ptr();
        let p2 = a2.as_ptr();
        let p3 = a3.as_ptr();
        let pq = q.as_ptr();

        let mut i = 0usize;
        while i + 4 <= N {
            let vq = vld1q_f32(pq.add(i));
            acc0 = vfmaq_f32(acc0, vld1q_f32(p0.add(i)), vq);
            acc1 = vfmaq_f32(acc1, vld1q_f32(p1.add(i)), vq);
            acc2 = vfmaq_f32(acc2, vld1q_f32(p2.add(i)), vq);
            acc3 = vfmaq_f32(acc3, vld1q_f32(p3.add(i)), vq);
            i += 4;
        }

        let mut r = [
            vaddvq_f32(acc0),
            vaddvq_f32(acc1),
            vaddvq_f32(acc2),
            vaddvq_f32(acc3),
        ];
        // Scalar tail for `N % 4 != 0` (never hit by our current dims —
        // 32/100/128/960 are all multiples of 4 — but kept for safety).
        while i < N {
            let qi = q[i];
            r[0] += a0[i] * qi;
            r[1] += a1[i] * qi;
            r[2] += a2[i] * qi;
            r[3] += a3[i] * qi;
            i += 1;
        }

        [-r[0], -r[1], -r[2], -r[3]]
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(never)]
pub fn distance_ip_vector_f32_batch4<const N: usize>(
    a0: &[f32; N],
    a1: &[f32; N],
    a2: &[f32; N],
    a3: &[f32; N],
    q: &[f32; N],
) -> [f32; 4] {
    [
        distance_ip_vector_f32::<N>(a0, q),
        distance_ip_vector_f32::<N>(a1, q),
        distance_ip_vector_f32::<N>(a2, q),
        distance_ip_vector_f32::<N>(a3, q),
    ]
}

