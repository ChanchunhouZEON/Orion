/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Runtime-dispatched x86 L2. CPU selection is cached once per scalar type.
//! Explicit AVX2/FMA builds bypass the function pointer; AVX512 builds retain
//! the separate compile-time kernels.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::Half;
#[cfg(not(all(
    target_feature = "avx2",
    target_feature = "fma",
    target_feature = "f16c"
)))]
use std::sync::OnceLock;

#[cfg(not(all(target_feature = "avx2", target_feature = "fma")))]
type F32Kernel = unsafe fn(&[f32], &[f32]) -> f32;
#[cfg(not(all(
    target_feature = "avx2",
    target_feature = "fma",
    target_feature = "f16c"
)))]
type F16Kernel = unsafe fn(&[Half], &[Half]) -> f32;

/// Exact squared L2 for equal-length slices, automatically using AVX2/FMA.
/// # Panics
/// Panics if lengths differ. Only naturally aligned slice storage is required.
#[inline]
pub fn distance_l2_slice_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(all(target_feature = "avx2", target_feature = "fma"))]
    {
        // SAFETY: the build target promises these CPU features; lengths match.
        unsafe { avx2_f32(a, b) }
    }
    #[cfg(not(all(target_feature = "avx2", target_feature = "fma")))]
    {
        static KERNEL: OnceLock<F32Kernel> = OnceLock::new();
        let kernel = KERNEL.get_or_init(|| {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                avx2_f32
            } else {
                scalar_f32
            }
        });
        // SAFETY: selection checks CPU support once; both kernels accept tails.
        unsafe { kernel(a, b) }
    }
}

#[inline]
pub fn distance_l2_vector_f32<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    distance_l2_slice_f32(a, b)
}

#[inline]
pub fn distance_l2_vector_f16<const N: usize>(a: &[Half; N], b: &[Half; N]) -> f32 {
    #[cfg(all(
        target_feature = "avx2",
        target_feature = "fma",
        target_feature = "f16c"
    ))]
    {
        // SAFETY: fixed-size arrays match and the build target promises support.
        unsafe { avx2_f16(a, b) }
    }
    #[cfg(not(all(
        target_feature = "avx2",
        target_feature = "fma",
        target_feature = "f16c"
    )))]
    {
        // A nongeneric slice signature is essential: a static inside a generic
        // function is shared across N, so it must never cache a kernel for one N.
        static KERNEL: OnceLock<F16Kernel> = OnceLock::new();
        let kernel = KERNEL.get_or_init(|| {
            if std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("fma")
                && std::is_x86_feature_detected!("f16c")
            {
                avx2_f16
            } else {
                scalar_f16
            }
        });
        // SAFETY: selection includes F16C, and both slices have length N.
        unsafe { kernel(a, b) }
    }
}

#[cfg(any(test, not(all(target_feature = "avx2", target_feature = "fma"))))]
fn scalar_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}
#[cfg(any(
    test,
    not(all(
        target_feature = "avx2",
        target_feature = "fma",
        target_feature = "f16c"
    ))
))]
fn scalar_f16(a: &[Half], b: &[Half]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x.to_f32() - y.to_f32();
            d * d
        })
        .sum()
}

// SAFETY contract: caller verifies AVX2/FMA and equal slice lengths. Loads are
// unaligned, each vector fits in-bounds, and the scalar loop handles the tail.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn avx2_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
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

// SAFETY contract: caller also verifies F16C; Half is repr(transparent) over f16.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn avx2_f16(a: &[Half], b: &[Half]) -> f32 {
    unsafe {
        let mut sum = _mm256_setzero_ps();
        let a_ptr = a.as_ptr() as *const __m128i;
        let b_ptr = b.as_ptr() as *const __m128i;

        for i in (0..a.len() / 8 * 8).step_by(8) {
            let a_vec = _mm256_cvtph_ps(_mm_loadu_si128(a_ptr.add(i / 8)));
            let b_vec = _mm256_cvtph_ps(_mm_loadu_si128(b_ptr.add(i / 8)));
            let diff = _mm256_sub_ps(a_vec, b_vec);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(sum, 1), _mm256_castps256_ps128(sum));
        let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        let mut total = _mm_cvtss_f32(x32);
        for i in a.len() / 8 * 8..a.len() {
            let diff = a[i].to_f32() - b[i].to_f32();
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
        assert_eq!(
            distance_l2_vector_f32(&a.values, &b.values),
            scalar_f32(&a.values, &b.values)
        );
        let a = OffsetHalf {
            pad: Half::from_f32(0.0),
            values: [Half::from_f32(1.0); N],
        };
        let b = OffsetHalf {
            pad: Half::from_f32(0.0),
            values: [Half::from_f32(3.0); N],
        };
        assert_ne!(a.values.as_ptr() as usize % 16, 0);
        assert_eq!(
            distance_l2_vector_f16(&a.values, &b.values),
            scalar_f16(&a.values, &b.values)
        );
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

    #[test]
    fn dispatch_matches_scalar_across_lengths_and_threads() {
        // Exercise the shared dispatch cache with changing lengths and callers.
        // Nonintegral data also covers floating-point reduction differences.
        std::thread::scope(|scope| {
            for offset in 0..4 {
                scope.spawn(move || {
                    for n in [1, 7, 8, 9, 31, 128, 129, 960] {
                        let a: Vec<f32> = (0..n).map(|i| (i + offset) as f32 / 17.0).collect();
                        let b: Vec<f32> = (0..n).map(|i| (i % 7) as f32 / 13.0).collect();
                        let expected = scalar_f32(&a, &b);
                        let actual = distance_l2_slice_f32(&a, &b);
                        assert!((actual - expected).abs() <= 1e-5 * expected.max(1.0));
                    }
                    check::<9>();
                    check::<25>();
                });
            }
        });
    }

    #[test]
    #[should_panic]
    fn mismatched_slices_fail_before_simd_loads() {
        distance_l2_slice_f32(&[1.0; 9], &[1.0; 8]);
    }
}
