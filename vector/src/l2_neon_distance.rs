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
/// Processes 4 floats at a time.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn distance_l2_vector_f32<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    debug_assert_eq!(N % 4, 0);

    unsafe {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // Unroll by 2 for better pipeline utilization
        let chunks = N / 8;
        for i in 0..chunks {
            let offset = i * 8;
            let a0 = vld1q_f32(a_ptr.add(offset));
            let b0 = vld1q_f32(b_ptr.add(offset));
            let diff0 = vsubq_f32(a0, b0);
            sum0 = vfmaq_f32(sum0, diff0, diff0);

            let a1 = vld1q_f32(a_ptr.add(offset + 4));
            let b1 = vld1q_f32(b_ptr.add(offset + 4));
            let diff1 = vsubq_f32(a1, b1);
            sum1 = vfmaq_f32(sum1, diff1, diff1);
        }

        // Handle remaining 4-element chunk
        let remaining_start = chunks * 8;
        if remaining_start < N {
            for i in (remaining_start..N).step_by(4) {
                let a_vec = vld1q_f32(a_ptr.add(i));
                let b_vec = vld1q_f32(b_ptr.add(i));
                let diff = vsubq_f32(a_vec, b_vec);
                sum0 = vfmaq_f32(sum0, diff, diff);
            }
        }

        let total = vaddq_f32(sum0, sum1);
        vaddvq_f32(total)
    }
}
