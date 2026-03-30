/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#![cfg_attr(
    not(test),
    warn(clippy::panic, clippy::unwrap_used, clippy::expect_used)
)]

mod distance;
mod half;
mod metric;
mod utils;
mod vector_storage;

// Architecture-specific SIMD distance implementations
#[cfg(target_arch = "x86_64")]
mod l2_float_distance;

#[cfg(target_arch = "aarch64")]
mod l2_neon_distance;

// Scalar fallback for architectures without SIMD support
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod l2_scalar_distance;

pub use crate::half::Half;
pub use distance::FullPrecisionDistance;
pub use metric::Metric;
pub use utils::prefetch_vector;
pub use vector_storage::VectorStorage;

#[cfg(test)]
mod test_util;

#[cfg(test)]
mod distance_test {
    use super::*;
    use crate::test_util::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dist_l2_f32_small() {
        // Use a dimension divisible by both 4 (NEON) and 8 (AVX2)
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let dist =
            <[f32; 8] as FullPrecisionDistance<f32, 8>>::distance_compare(&a, &b, Metric::L2);
        // Each dimension differs by 1.0, so L2 = 8 * 1.0^2 = 8.0
        assert_abs_diff_eq!(dist, 8.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dist_l2_f32_zeros() {
        let a = [0.0f32; 8];
        let b = [0.0f32; 8];
        let dist =
            <[f32; 8] as FullPrecisionDistance<f32, 8>>::distance_compare(&a, &b, Metric::L2);
        assert_abs_diff_eq!(dist, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dist_l2_f32_matches_scalar() {
        let a: [f32; 16] = [
            1.5, 2.3, -0.7, 4.1, 0.0, -3.2, 8.8, 1.0, -2.5, 6.7, 0.1, -1.1, 3.3, 9.9, -4.4, 2.2,
        ];
        let b: [f32; 16] = [
            -0.5, 1.1, 2.3, -1.0, 5.5, 0.8, -2.1, 3.7, 4.4, -0.3, 7.7, 2.2, -1.8, 0.6, 5.5, -3.3,
        ];

        let simd_dist =
            <[f32; 16] as FullPrecisionDistance<f32, 16>>::distance_compare(&a, &b, Metric::L2);
        let scalar_dist = no_vector_compare_f32(&a, &b);
        assert_abs_diff_eq!(simd_dist, scalar_dist, epsilon = 1e-4);
    }

    #[test]
    fn test_dist_l2_f16_matches_scalar() {
        let a_f32: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b_f32: [f32; 8] = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let a: [Half; 8] = std::array::from_fn(|i| Half::from_f32(a_f32[i]));
        let b: [Half; 8] = std::array::from_fn(|i| Half::from_f32(b_f32[i]));

        let dist =
            <[Half; 8] as FullPrecisionDistance<Half, 8>>::distance_compare(&a, &b, Metric::L2);
        let scalar_dist = no_vector_compare_f16(&a, &b);
        assert_eq!(dist, scalar_dist);
    }
}
