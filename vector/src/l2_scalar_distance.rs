/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Scalar fallback L2 distance computation for non-SIMD architectures.

use crate::Half;

/// Calculate L2 squared distance between two f16 vectors (scalar).
#[inline(never)]
pub fn distance_l2_vector_f16<const N: usize>(a: &[Half; N], b: &[Half; N]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..N {
        let diff = a[i].to_f32() - b[i].to_f32();
        sum += diff * diff;
    }
    sum
}

/// Calculate L2 squared distance between two f32 vectors (scalar).
#[inline(never)]
pub fn distance_l2_vector_f32<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..N {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}
