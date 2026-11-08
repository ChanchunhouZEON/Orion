/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#[cfg(test)]
use crate::Half;

#[cfg(test)]
pub fn no_vector_compare_f16(a: &[Half], b: &[Half]) -> f32 {
    let mut sum = 0.0;
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        sum += (a[i].to_f32() - b[i].to_f32()).powi(2);
    }
    sum
}

#[cfg(test)]
pub fn no_vector_compare_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        sum += (a[i] - b[i]).powi(2);
    }
    sum
}

#[cfg(test)]
pub fn scalar_ip_f32(a: &[f32], b: &[f32]) -> f32 {
    // Returns `-Σ a·b` ("smaller == closer" — matches `distance_ip_*`).
    debug_assert_eq!(a.len(), b.len());
    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

#[cfg(test)]
pub fn scalar_l2_u8(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum: u32 = 0;
    for i in 0..a.len() {
        let d = (a[i] as i32 - b[i] as i32).unsigned_abs();
        sum += d * d;
    }
    sum as f32
}

#[cfg(test)]
pub fn scalar_ip_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let s: i32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as i32) * (*y as i32))
        .sum();
    -s
}

#[cfg(test)]
pub fn scalar_ip_i16(a: &[i16], b: &[i16]) -> i64 {
    debug_assert_eq!(a.len(), b.len());
    let s: i64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as i64) * (*y as i64))
        .sum();
    -s
}
