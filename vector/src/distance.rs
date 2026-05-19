/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::Half;
use crate::Metric;

// Architecture-specific distance imports
#[cfg(target_arch = "x86_64")]
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
        #[cfg(not(target_arch = "aarch64"))]
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
        #[cfg(not(target_arch = "aarch64"))]
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
    fn distance_compare(_a: &[u8; N], _b: &[u8; N], _metric: Metric) -> f32 {
        panic!("Not supported VectorType u8")
    }
}
