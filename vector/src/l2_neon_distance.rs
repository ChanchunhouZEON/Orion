/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! NEON L2 distance computation for aarch64.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::Half;

/// Calculate L2 squared distance between two f16 vectors using NEON.
/// Processes 4 floats at a time (vs AVX2's 8).
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn distance_l2_vector_f16<const N: usize>(a: &[Half; N], b: &[Half; N]) -> f32 {
    debug_assert_eq!(N % 4, 0);

    unsafe {
        let mut sum = vdupq_n_f32(0.0);

        // f16 -> f32 conversion then NEON distance
        // Process 4 f16 values at a time by converting to f32 first
        let a_f32: Vec<f32> = (0..N)
            .map(|i| {
                let half_ptr = a.as_ptr().add(i) as *const Half;
                (*half_ptr).to_f32()
            })
            .collect();
        let b_f32: Vec<f32> = (0..N)
            .map(|i| {
                let half_ptr = b.as_ptr().add(i) as *const Half;
                (*half_ptr).to_f32()
            })
            .collect();

        let a_ptr = a_f32.as_ptr();
        let b_ptr = b_f32.as_ptr();

        for i in (0..N).step_by(4) {
            let a_vec = vld1q_f32(a_ptr.add(i));
            let b_vec = vld1q_f32(b_ptr.add(i));
            let diff = vsubq_f32(a_vec, b_vec);
            sum = vfmaq_f32(sum, diff, diff);
        }

        vaddvq_f32(sum)
    }
}

/// Calculate L2 squared distance between two f32 vectors using NEON.
/// 4-way unrolled: processes 16 floats per iteration to reduce loop overhead.
/// Both input arrays MUST be 16-byte aligned (guaranteed by AlignedBoxWithSlice
/// for dataset vectors and AlignedQuery for query vectors).
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn distance_l2_vector_f32<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    debug_assert_eq!(N % 4, 0);

    unsafe {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // Prefetch distance: 4 iterations ahead = 256 bytes per stream.
        // On M2 (128B cache line), this is 2 cache lines — enough to hide
        // ~60 cycle L2 latency at 4 NEON ops/cycle.
        const PF_AHEAD: usize = 4;

        // Unroll by 4: process 16 floats per iteration
        let chunks = N / 16;
        for i in 0..chunks {
            let offset = i * 16;

            // Prefetch future data for both streams
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
            let diff0 = vsubq_f32(a0, b0);
            sum0 = vfmaq_f32(sum0, diff0, diff0);

            let a1 = vld1q_f32(a_ptr.add(offset + 4));
            let b1 = vld1q_f32(b_ptr.add(offset + 4));
            let diff1 = vsubq_f32(a1, b1);
            sum1 = vfmaq_f32(sum1, diff1, diff1);

            let a2 = vld1q_f32(a_ptr.add(offset + 8));
            let b2 = vld1q_f32(b_ptr.add(offset + 8));
            let diff2 = vsubq_f32(a2, b2);
            sum2 = vfmaq_f32(sum2, diff2, diff2);

            let a3 = vld1q_f32(a_ptr.add(offset + 12));
            let b3 = vld1q_f32(b_ptr.add(offset + 12));
            let diff3 = vsubq_f32(a3, b3);
            sum3 = vfmaq_f32(sum3, diff3, diff3);
        }

        // Handle remaining elements in 4-float steps
        let remaining_start = chunks * 16;
        for i in (remaining_start..N).step_by(4) {
            let a_vec = vld1q_f32(a_ptr.add(i));
            let b_vec = vld1q_f32(b_ptr.add(i));
            let diff = vsubq_f32(a_vec, b_vec);
            sum0 = vfmaq_f32(sum0, diff, diff);
        }

        let total = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
        vaddvq_f32(total)
    }
}

/// L2 squared distance with early abandon.
/// Returns the actual distance if < `upper_bound`, or -1.0 if partial
/// distance exceeds `upper_bound` before all dimensions are processed.
/// Checks every 128 dimensions (8 unrolled iterations).
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn distance_l2_early_abandon_f32<const N: usize>(
    a: &[f32; N],
    b: &[f32; N],
    upper_bound: f32,
) -> f32 {
    debug_assert_eq!(N % 4, 0);

    unsafe {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        const PF_AHEAD: usize = 4;
        // Check every 8 iterations = 128 floats = 512 bytes per stream
        const CHECK_INTERVAL: usize = 8;

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
            let diff0 = vsubq_f32(a0, b0);
            sum0 = vfmaq_f32(sum0, diff0, diff0);

            let a1 = vld1q_f32(a_ptr.add(offset + 4));
            let b1 = vld1q_f32(b_ptr.add(offset + 4));
            let diff1 = vsubq_f32(a1, b1);
            sum1 = vfmaq_f32(sum1, diff1, diff1);

            let a2 = vld1q_f32(a_ptr.add(offset + 8));
            let b2 = vld1q_f32(b_ptr.add(offset + 8));
            let diff2 = vsubq_f32(a2, b2);
            sum2 = vfmaq_f32(sum2, diff2, diff2);

            let a3 = vld1q_f32(a_ptr.add(offset + 12));
            let b3 = vld1q_f32(b_ptr.add(offset + 12));
            let diff3 = vsubq_f32(a3, b3);
            sum3 = vfmaq_f32(sum3, diff3, diff3);

            // Check every 128 dims: abandon if partial sum >= upper_bound
            if (i + 1) % CHECK_INTERVAL == 0 {
                let partial = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
                if vaddvq_f32(partial) >= upper_bound {
                    return -1.0;
                }
            }
        }

        // Remaining
        let remaining_start = chunks * 16;
        for i in (remaining_start..N).step_by(4) {
            let a_vec = vld1q_f32(a_ptr.add(i));
            let b_vec = vld1q_f32(b_ptr.add(i));
            let diff = vsubq_f32(a_vec, b_vec);
            sum0 = vfmaq_f32(sum0, diff, diff);
        }

        let total = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
        let dist = vaddvq_f32(total);
        if dist >= upper_bound { -1.0 } else { dist }
    }
}
