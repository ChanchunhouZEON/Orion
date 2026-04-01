/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use ndarray::ArrayView1;

/// L2 distance between two ndarray row views.
/// Converts to contiguous slices and dispatches to the SIMD path.
#[inline]
pub fn l2_distance(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    match (a.as_slice(), b.as_slice()) {
        (Some(a_s), Some(b_s)) => l2_distance_slice(a_s, b_s),
        // Non-contiguous fallback (rare for row-major ArcArray2)
        _ => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt(),
    }
}

/// L2 distance between two f32 slices.
/// Dispatches to SIMD (NEON on aarch64, AVX2 on x86_64) or scalar fallback.
#[inline]
pub fn l2_distance_slice(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    l2_sq_slice(a, b).sqrt()
}

// ─── Internal: squared L2 dispatcher ────────────────────────────────────────

#[inline]
fn l2_sq_slice(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64 (Apple Silicon, AWS Graviton, etc.)
        // Safety: NEON guaranteed on this target.
        unsafe { l2_sq_neon(a, b) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Runtime check: AVX2 + FMA are common on modern x86 but not universal.
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // Safety: we just verified the CPU features are present.
            unsafe { l2_sq_avx2(a, b) }
        } else {
            l2_sq_scalar(a, b)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        l2_sq_scalar(a, b)
    }
}

// ─── aarch64 / NEON ─────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn l2_sq_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // Two independent accumulators for out-of-order execution / ILP.
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);

    // Main loop: 8 elements per iteration (2 × float32x4), unrolled 2-way.
    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let off = i * 8;
        let a0 = vld1q_f32(a_ptr.add(off));
        let b0 = vld1q_f32(b_ptr.add(off));
        let d0 = vsubq_f32(a0, b0);
        sum0 = vfmaq_f32(sum0, d0, d0);

        let a1 = vld1q_f32(a_ptr.add(off + 4));
        let b1 = vld1q_f32(b_ptr.add(off + 4));
        let d1 = vsubq_f32(a1, b1);
        sum1 = vfmaq_f32(sum1, d1, d1);
    }

    // Remaining 4-element chunk (if any).
    let rem4_start = chunks8 * 8;
    if rem4_start + 4 <= n {
        let a_v = vld1q_f32(a_ptr.add(rem4_start));
        let b_v = vld1q_f32(b_ptr.add(rem4_start));
        let dv = vsubq_f32(a_v, b_v);
        sum0 = vfmaq_f32(sum0, dv, dv);
    }

    // Horizontal reduction.
    let mut result = vaddvq_f32(vaddq_f32(sum0, sum1));

    // Scalar tail (< 4 remaining elements).
    let tail_start = n & !3; // round down to multiple of 4
    for i in tail_start..n {
        let d = *a_ptr.add(i) - *b_ptr.add(i);
        result += d * d;
    }

    result
}

// ─── x86_64 / AVX2 + FMA ────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn l2_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();

    // Main loop: 16 elements per iteration (2 × __m256), unrolled 2-way.
    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let off = i * 16;
        let a0 = _mm256_loadu_ps(a_ptr.add(off));
        let b0 = _mm256_loadu_ps(b_ptr.add(off));
        let d0 = _mm256_sub_ps(a0, b0);
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);

        let a1 = _mm256_loadu_ps(a_ptr.add(off + 8));
        let b1 = _mm256_loadu_ps(b_ptr.add(off + 8));
        let d1 = _mm256_sub_ps(a1, b1);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);
    }

    // Remaining 8-element chunk (if any).
    let rem8_start = chunks16 * 16;
    if rem8_start + 8 <= n {
        let a_v = _mm256_loadu_ps(a_ptr.add(rem8_start));
        let b_v = _mm256_loadu_ps(b_ptr.add(rem8_start));
        let dv = _mm256_sub_ps(a_v, b_v);
        sum0 = _mm256_fmadd_ps(dv, dv, sum0);
    }

    // Horizontal reduction: collapse 8 lanes to 1 scalar.
    let combined = _mm256_add_ps(sum0, sum1);
    let lo128 = _mm256_castps256_ps128(combined);
    let hi128 = _mm256_extractf128_ps(combined, 1);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, shuf2);
    let mut result = _mm_cvtss_f32(sum32);

    // Scalar tail (< 8 remaining elements).
    let tail_start = n & !7; // round down to multiple of 8
    for i in tail_start..n {
        let d = *a_ptr.add(i) - *b_ptr.add(i);
        result += d * d;
    }

    result
}

// ─── Scalar fallback ─────────────────────────────────────────────────────────

#[allow(dead_code)]
#[inline]
fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn test_l2_distance_slice_128() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| (128 - i) as f32 * 0.1).collect();
        let expected = reference_l2(&a, &b);
        let got = l2_distance_slice(&a, &b);
        assert!(
            (got - expected).abs() < 1e-3,
            "dim=128: expected {expected}, got {got}"
        );
    }

    #[test]
    fn test_l2_distance_slice_non_multiple() {
        // Length not a multiple of 4 or 8
        let a: Vec<f32> = (0..17).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..17).map(|i| (17 - i) as f32).collect();
        let expected = reference_l2(&a, &b);
        let got = l2_distance_slice(&a, &b);
        assert!(
            (got - expected).abs() < 1e-3,
            "dim=17: expected {expected}, got {got}"
        );
    }

    #[test]
    fn test_l2_distance_slice_identical() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!((l2_distance_slice(&a, &a) - 0.0).abs() < 1e-6);
    }
}
