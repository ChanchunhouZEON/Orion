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
