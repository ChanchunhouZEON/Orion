/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::Half;
use crate::Metric;

// Architecture-specific distance imports — order matches the lib.rs
// dispatch (AVX-512 > AVX2 > NEON > scalar, picking the first that
// the target satisfies).
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use crate::l2_avx512_distance::{distance_l2_vector_f16, distance_l2_vector_f32};

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
use crate::l2_float_distance::{distance_l2_vector_f16, distance_l2_vector_f32};

#[cfg(target_arch = "aarch64")]
use crate::l2_neon_distance::{distance_l2_vector_f16, distance_l2_vector_f32};

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
use crate::l2_scalar_distance::{distance_l2_vector_f16, distance_l2_vector_f32};

/// Distance contract for full-precision vertex
pub trait FullPrecisionDistance<T, const N: usize> {
    /// Get the distance between vertex a and vertex b
    fn distance_compare(a: &[T; N], b: &[T; N], vec_type: Metric) -> f32;

    /// L2 distance with early abandon: returns the actual distance if it is
    /// less than `upper_bound`, or a negative value (-1.0) if the partial
    /// distance already exceeds `upper_bound` before all dimensions are
    /// processed. Callers check `result < 0.0` to detect abandonment.
    fn distance_compare_with_bound(a: &[T; N], b: &[T; N], upper_bound: f32) -> f32 {
        let d = Self::distance_compare(a, b, Metric::L2);
        if d < upper_bound {
            d
        } else {
            -1.0
        }
    }

    /// ADSampling L2 distance: assumes both vectors have been pre-rotated by a
    /// random orthogonal matrix. Abandons early when the scaled partial sum
    /// exceeds `upper_bound × (1 + ε/√d')`. Returns `-1.0` on abandonment.
    ///
    /// Default impl falls back to the rotation-free bounded compare — only
    /// the `[f32; N]` specialization actually does scaled early abort.
    fn distance_compare_adsampling(a: &[T; N], b: &[T; N], upper_bound: f32, _epsilon: f32) -> f32 {
        Self::distance_compare_with_bound(a, b, upper_bound)
    }
}

#[allow(clippy::panic)]
impl<const N: usize> FullPrecisionDistance<f32, N> for [f32; N] {
    #[inline(always)]
    fn distance_compare(a: &[f32; N], b: &[f32; N], metric: Metric) -> f32 {
        match metric {
            Metric::L2 => distance_l2_vector_f32::<N>(a, b),
            _ => panic!("Not supported Metric type {:?}", metric),
        }
    }

    #[inline(always)]
    fn distance_compare_with_bound(a: &[f32; N], b: &[f32; N], upper_bound: f32) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            crate::l2_neon_distance::distance_l2_early_abandon_f32::<N>(a, b, upper_bound)
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            crate::l2_avx512_distance::distance_l2_early_abandon_f32::<N>(a, b, upper_bound)
        }
        #[cfg(not(any(
            target_arch = "aarch64",
            all(target_arch = "x86_64", target_feature = "avx512f")
        )))]
        {
            let d = distance_l2_vector_f32::<N>(a, b);
            if d < upper_bound {
                d
            } else {
                -1.0
            }
        }
    }

    #[inline(always)]
    fn distance_compare_adsampling(
        a: &[f32; N],
        b: &[f32; N],
        upper_bound: f32,
        epsilon: f32,
    ) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            crate::l2_neon_distance::distance_l2_adsampling_f32::<N>(a, b, upper_bound, epsilon)
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            crate::l2_avx512_distance::distance_l2_adsampling_f32::<N>(a, b, upper_bound, epsilon)
        }
        #[cfg(not(any(
            target_arch = "aarch64",
            all(target_arch = "x86_64", target_feature = "avx512f")
        )))]
        {
            // Scalar fallback: scaled-threshold early abort in chunks of 32 dims.
            let n_f = N as f32;
            const CHECK_EVERY: usize = 32;
            let mut sum = 0.0f32;
            let mut processed = 0usize;
            for i in 0..N {
                let d = a[i] - b[i];
                sum += d * d;
                processed += 1;
                if processed % CHECK_EVERY == 0 && processed < N {
                    let lhs = sum * (n_f / processed as f32);
                    let rhs = upper_bound * (1.0 + epsilon / (processed as f32).sqrt());
                    if lhs > rhs {
                        return -1.0;
                    }
                }
            }
            if sum >= upper_bound {
                -1.0
            } else {
                sum
            }
        }
    }
}

#[allow(clippy::panic)]
impl<const N: usize> FullPrecisionDistance<Half, N> for [Half; N] {
    fn distance_compare(a: &[Half; N], b: &[Half; N], metric: Metric) -> f32 {
        match metric {
            Metric::L2 => distance_l2_vector_f16::<N>(a, b),
            _ => panic!("Not supported Metric type {:?}", metric),
        }
    }
}

#[allow(clippy::panic)]
impl<const N: usize> FullPrecisionDistance<i8, N> for [i8; N] {
    fn distance_compare(_a: &[i8; N], _b: &[i8; N], _metric: Metric) -> f32 {
        panic!("Not supported VectorType i8")
    }
}

#[allow(clippy::panic)]
impl<const N: usize> FullPrecisionDistance<u8, N> for [u8; N] {
    /// Squared L2 on original byte coordinates, without quantization or scaling.
    /// For SIFT's 128 dimensions even the maximum sum is below 2^24, so the
    /// integer result is represented exactly by the returned f32.
    #[inline(always)]
    fn distance_compare(a: &[u8; N], b: &[u8; N], metric: Metric) -> f32 {
        if metric != Metric::L2 {
            panic!("Not supported Metric type {:?} for u8", metric);
        }

        // Existing SIMD kernels consume fixed blocks and use 32-bit accumulators.
        // A multiple of 32 avoids partial-block loads on both NEON and AVX-512;
        // the dimension cap keeps even the signed x86 reduction below i32::MAX.
        if N % 32 == 0 && N <= 32_768 {
            crate::distance_l2_vector_u8::<N>(a, b)
        } else {
            // FullPrecisionDistance accepts arbitrary N. Widen before subtraction,
            // and sum in u64 so large dimensions cannot wrap a 32-bit accumulator.
            let mut sum = 0u64;
            for (&left, &right) in a.iter().zip(b) {
                let difference = i32::from(left) - i32::from(right);
                sum += (difference * difference) as u64;
            }
            sum as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Half;

    #[test]
    fn distance_compare_with_bound_returns_distance_when_under() {
        let a = [0.0f32; 8];
        let b = [1.0f32; 8];
        // True L2 = 8.0; bound > 8 → returns 8.0
        let d =
            <[f32; 8] as FullPrecisionDistance<f32, 8>>::distance_compare_with_bound(&a, &b, 100.0);
        assert!(d > 0.0 && d < 100.0);
    }

    #[test]
    fn distance_compare_with_bound_returns_negative_when_over() {
        let a = [0.0f32; 8];
        let b = [10.0f32; 8];
        // True L2 = 800.0; bound = 1.0 → must signal abandon (< 0).
        let d =
            <[f32; 8] as FullPrecisionDistance<f32, 8>>::distance_compare_with_bound(&a, &b, 1.0);
        assert!(d < 0.0);
    }

    #[test]
    fn distance_compare_adsampling_full_value_when_unbound() {
        let a: [f32; 64] = std::array::from_fn(|i| i as f32 * 0.1);
        let b: [f32; 64] = std::array::from_fn(|i| (i as f32 * 0.1) + 0.5);
        // Large upper bound → no abandon → returns the full distance.
        let d = <[f32; 64] as FullPrecisionDistance<f32, 64>>::distance_compare_adsampling(
            &a, &b, 1e9, 2.1,
        );
        assert!(d > 0.0);
    }

    #[test]
    fn half_l2_metric_dispatch() {
        let a: [Half; 4] = std::array::from_fn(|i| Half::from_f32(i as f32));
        let b: [Half; 4] = std::array::from_fn(|i| Half::from_f32((i as f32) + 1.0));
        let d = <[Half; 4] as FullPrecisionDistance<Half, 4>>::distance_compare(&a, &b, Metric::L2);
        assert!((d - 4.0).abs() < 1e-3);
    }

    #[test]
    #[should_panic(expected = "Not supported")]
    fn half_cosine_metric_panics() {
        let a: [Half; 4] = std::array::from_fn(|_| Half::from_f32(0.0));
        <[Half; 4] as FullPrecisionDistance<Half, 4>>::distance_compare(&a, &a, Metric::Cosine);
    }

    #[test]
    #[should_panic(expected = "VectorType i8")]
    fn i8_storage_panics() {
        let a = [0i8; 4];
        <[i8; 4] as FullPrecisionDistance<i8, 4>>::distance_compare(&a, &a, Metric::L2);
    }

    fn check_u8_against_integer_oracle<const N: usize>() {
        let a = std::array::from_fn(|i| (i.wrapping_mul(73).wrapping_add(255)) as u8);
        let b = std::array::from_fn(|i| (i.wrapping_mul(151).wrapping_add(17)) as u8);
        let expected: u64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| {
                let difference = i64::from(x) - i64::from(y);
                (difference * difference) as u64
            })
            .sum();
        let compare = <[u8; N] as FullPrecisionDistance<u8, N>>::distance_compare;
        assert_eq!(compare(&a, &b, Metric::L2), expected as f32);
        assert_eq!(compare(&b, &a, Metric::L2), expected as f32);
        assert_eq!(compare(&a, &a, Metric::L2), 0.0);
    }

    #[test]
    fn u8_l2_handles_simd_blocks_and_arbitrary_tails() {
        check_u8_against_integer_oracle::<0>();
        check_u8_against_integer_oracle::<1>();
        check_u8_against_integer_oracle::<15>();
        check_u8_against_integer_oracle::<16>();
        check_u8_against_integer_oracle::<17>();
        check_u8_against_integer_oracle::<31>();
        check_u8_against_integer_oracle::<32>();
        check_u8_against_integer_oracle::<33>();
        check_u8_against_integer_oracle::<128>();
        check_u8_against_integer_oracle::<784>();
    }

    #[test]
    fn u8_sift_extremes_equal_full_precision_f32() {
        let a = [0u8; 128];
        let b = [255u8; 128];
        let byte_distance =
            <[u8; 128] as FullPrecisionDistance<u8, 128>>::distance_compare(&a, &b, Metric::L2);
        let float_distance = <[f32; 128] as FullPrecisionDistance<f32, 128>>::distance_compare(
            &a.map(f32::from),
            &b.map(f32::from),
            Metric::L2,
        );
        assert_eq!(byte_distance, 8_323_200.0);
        assert_eq!(byte_distance, float_distance);
    }

    #[test]
    fn u8_large_dimension_does_not_overflow_u32() {
        let distance = <[u8; 70_000] as FullPrecisionDistance<u8, 70_000>>::distance_compare(
            &[0; 70_000],
            &[255; 70_000],
            Metric::L2,
        );
        assert_eq!(distance, (70_000u64 * 65_025) as f32);
    }

    #[test]
    fn u8_bound_and_adsampling_fallback_follow_the_trait_contract() {
        type Distance = [u8; 128];
        let a = [0; 128];
        let b = [1; 128];
        assert_eq!(Distance::distance_compare_with_bound(&a, &b, 129.0), 128.0);
        assert_eq!(Distance::distance_compare_with_bound(&a, &b, 128.0), -1.0);
        assert_eq!(Distance::distance_compare_with_bound(&a, &b, 1.0), -1.0);
        assert_eq!(
            Distance::distance_compare_adsampling(&a, &b, 129.0, 2.1),
            128.0
        );
    }

    #[test]
    #[should_panic(expected = "Not supported Metric")]
    fn u8_cosine_is_not_silently_treated_as_l2() {
        <[u8; 4] as FullPrecisionDistance<u8, 4>>::distance_compare(
            &[0; 4],
            &[1; 4],
            Metric::Cosine,
        );
    }
}
