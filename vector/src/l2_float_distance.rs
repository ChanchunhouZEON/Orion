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
    debug_assert_eq!(N % 8, 0);
    debug_assert_eq!(a.as_ptr().align_offset(32), 0);
    debug_assert_eq!(b.as_ptr().align_offset(32), 0);

    unsafe {
        let mut sum = _mm256_setzero_ps();
        let a_ptr = a.as_ptr() as *const __m128i;
        let b_ptr = b.as_ptr() as *const __m128i;

        for i in (0..N).step_by(8) {
            let a_vec = _mm256_cvtph_ps(_mm_load_si128(a_ptr.add(i / 8)));
            let b_vec = _mm256_cvtph_ps(_mm_load_si128(b_ptr.add(i / 8)));
            let diff = _mm256_sub_ps(a_vec, b_vec);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(sum, 1), _mm256_castps256_ps128(sum));
        let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        _mm_cvtss_f32(x32)
    }
}

/// Calculate L2 squared distance between two f32 vectors using AVX2.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub fn distance_l2_vector_f32<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    debug_assert_eq!(N % 8, 0);
    debug_assert_eq!(a.as_ptr().align_offset(32), 0);
    debug_assert_eq!(b.as_ptr().align_offset(32), 0);

    unsafe {
        let mut sum = _mm256_setzero_ps();

        for i in (0..N).step_by(8) {
            let a_vec = _mm256_load_ps(&a[i]);
            let b_vec = _mm256_load_ps(&b[i]);
            let diff = _mm256_sub_ps(a_vec, b_vec);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(sum, 1), _mm256_castps256_ps128(sum));
        let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        _mm_cvtss_f32(x32)
    }
}
