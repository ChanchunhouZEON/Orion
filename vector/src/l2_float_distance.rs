/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! AVX2 L2 distance computation for x86_64.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::Half;

/// Calculate L2 squared distance between two f16 vectors using AVX2.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub fn distance_l2_vector_f16<const N: usize>(a: &[Half; N], b: &[Half; N]) -> f32 {
    // Array references guarantee element alignment only. Load unaligned and
    // stop before the tail so this safe API also accepts arbitrary dimensions.

    unsafe {
        let mut sum = _mm256_setzero_ps();
        let a_ptr = a.as_ptr() as *const __m128i;
        let b_ptr = b.as_ptr() as *const __m128i;

        for i in (0..N / 8 * 8).step_by(8) {
            let a_vec = _mm256_cvtph_ps(_mm_loadu_si128(a_ptr.add(i / 8)));
            let b_vec = _mm256_cvtph_ps(_mm_loadu_si128(b_ptr.add(i / 8)));
            let diff = _mm256_sub_ps(a_vec, b_vec);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(sum, 1), _mm256_castps256_ps128(sum));
        let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        let mut total = _mm_cvtss_f32(x32);
        for i in N / 8 * 8..N {
            let diff = a[i].to_f32() - b[i].to_f32();
            total += diff * diff;
        }
        total
    }
}

/// Calculate L2 squared distance between two f32 vectors using AVX2.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub fn distance_l2_vector_f32<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    distance_l2_slice_f32(a, b)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn distance_l2_slice_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    // Array references guarantee element alignment only. Load unaligned and
    // stop before the tail so this safe API also accepts arbitrary dimensions.

    unsafe {
        let mut sum = _mm256_setzero_ps();

        for i in (0..n / 8 * 8).step_by(8) {
            let a_vec = _mm256_loadu_ps(&a[i]);
            let b_vec = _mm256_loadu_ps(&b[i]);
            let diff = _mm256_sub_ps(a_vec, b_vec);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(sum, 1), _mm256_castps256_ps128(sum));
        let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        let mut total = _mm_cvtss_f32(x32);
        for i in n / 8 * 8..n {
            let diff = a[i] - b[i];
            total += diff * diff;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Force a known SIMD-misaligned address rather than relying on whichever
    // stack layout the optimizer happens to choose for ordinary arrays.
    #[repr(C, align(64))]
    struct OffsetF32<const N: usize> {
        pad: f32,
        values: [f32; N],
    }
    #[repr(C, align(64))]
    struct OffsetHalf<const N: usize> {
        pad: Half,
        values: [Half; N],
    }

    fn check<const N: usize>() {
        let a = OffsetF32 {
            pad: 0.0,
            values: [1.0; N],
        };
        let b = OffsetF32 {
            pad: 0.0,
            values: [3.0; N],
        };
        assert_ne!(a.values.as_ptr() as usize % 32, 0);
        assert_eq!(distance_l2_vector_f32(&a.values, &b.values), 4.0 * N as f32);
        let a = OffsetHalf {
            pad: Half::from_f32(0.0),
            values: [Half::from_f32(1.0); N],
        };
        let b = OffsetHalf {
            pad: Half::from_f32(0.0),
            values: [Half::from_f32(3.0); N],
        };
        assert_ne!(a.values.as_ptr() as usize % 16, 0);
        assert_eq!(distance_l2_vector_f16(&a.values, &b.values), 4.0 * N as f32);
    }

    #[test]
    fn unaligned_inputs_and_short_tails() {
        check::<0>();
        check::<1>();
        check::<4>();
        check::<8>();
        check::<9>();
        check::<25>();
        check::<100>();
        check::<128>();
    }
}
