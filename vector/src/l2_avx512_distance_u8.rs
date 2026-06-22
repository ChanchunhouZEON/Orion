/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! AVX-512 squared-L2 for u8 vectors. NEON counterpart in
//! `l2_neon_distance_u8.rs` — see that file's docstring for the
//! quantized-prefilter rationale.
//!
//! AVX-512 pipeline:
//!   * Load 32 u8 bytes per inner iter into a `__m256i`.
//!   * Widen to 32× i16 via `_mm512_cvtepu8_epi16`.
//!   * Subtract a − b → 32× i16 diff.
//!   * `_mm512_madd_epi16(diff, diff)` → 16× i32 squared-pair sums
//!     (one VPMADDWD), accumulated into a 512-bit i32 register.
//!
//! 4-way unroll lands at 128 u8 elements per outer iter.

#![cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]

use std::arch::x86_64::*;

#[inline(always)]
unsafe fn reduce_add_epi32(v: __m512i) -> i64 {
    _mm512_reduce_add_epi32(v) as i64
}

/// Squared-L2 on u8 vectors; returns the raw integer sum
/// `Σ (a_i - b_i)²` as f32. `N` must be a multiple of 16 to match
/// the NEON contract.
#[inline(never)]
pub fn distance_l2_vector_u8<const N: usize>(a: &[u8; N], b: &[u8; N]) -> f32 {
    debug_assert_eq!(N % 16, 0);
    unsafe {
        let mut acc0 = _mm512_setzero_si512();
        let mut acc1 = _mm512_setzero_si512();
        let mut acc2 = _mm512_setzero_si512();
        let mut acc3 = _mm512_setzero_si512();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // Outer: 128 u8 elements per iter (4 × 32-byte chunks).
        let chunks = N / 128;
        for i in 0..chunks {
            let off = i * 128;

            let va0 = _mm256_loadu_si256(a_ptr.add(off) as *const __m256i);
            let vb0 = _mm256_loadu_si256(b_ptr.add(off) as *const __m256i);
            let a0 = _mm512_cvtepu8_epi16(va0);
            let b0 = _mm512_cvtepu8_epi16(vb0);
            let d0 = _mm512_sub_epi16(a0, b0);
            acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(d0, d0));

            let va1 = _mm256_loadu_si256(a_ptr.add(off + 32) as *const __m256i);
            let vb1 = _mm256_loadu_si256(b_ptr.add(off + 32) as *const __m256i);
            let a1 = _mm512_cvtepu8_epi16(va1);
            let b1 = _mm512_cvtepu8_epi16(vb1);
            let d1 = _mm512_sub_epi16(a1, b1);
            acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(d1, d1));

            let va2 = _mm256_loadu_si256(a_ptr.add(off + 64) as *const __m256i);
            let vb2 = _mm256_loadu_si256(b_ptr.add(off + 64) as *const __m256i);
            let a2 = _mm512_cvtepu8_epi16(va2);
            let b2 = _mm512_cvtepu8_epi16(vb2);
            let d2 = _mm512_sub_epi16(a2, b2);
            acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(d2, d2));

            let va3 = _mm256_loadu_si256(a_ptr.add(off + 96) as *const __m256i);
            let vb3 = _mm256_loadu_si256(b_ptr.add(off + 96) as *const __m256i);
            let a3 = _mm512_cvtepu8_epi16(va3);
            let b3 = _mm512_cvtepu8_epi16(vb3);
            let d3 = _mm512_sub_epi16(a3, b3);
            acc3 = _mm512_add_epi32(acc3, _mm512_madd_epi16(d3, d3));
        }

        // Tail: 32-element strides for the remainder.
        let mut i = chunks * 128;
        while i + 32 <= N {
            let va = _mm256_loadu_si256(a_ptr.add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b_ptr.add(i) as *const __m256i);
            let aw = _mm512_cvtepu8_epi16(va);
            let bw = _mm512_cvtepu8_epi16(vb);
            let d = _mm512_sub_epi16(aw, bw);
            acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(d, d));
            i += 32;
        }
        // Last 16-byte chunk (N is a multiple of 16, so the only
        // possible residue is exactly 16 elements).
        if i + 16 <= N {
            let va = _mm_loadu_si128(a_ptr.add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b_ptr.add(i) as *const __m128i);
            let aw = _mm256_cvtepu8_epi16(va);
            let bw = _mm256_cvtepu8_epi16(vb);
            let d = _mm256_sub_epi16(aw, bw);
            // Widen the 256-bit madd result and add into acc0's
            // low half. `_mm512_castsi256_si512` zero-extends.
            let m = _mm256_madd_epi16(d, d);
            acc0 = _mm512_add_epi32(acc0, _mm512_castsi256_si512(m));
        }

        let acc = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1), _mm512_add_epi32(acc2, acc3));
        reduce_add_epi32(acc) as f32
    }
}

/// Batched u8 L2: 4 candidates × 1 query in a single pass —
/// mirrors NEON's `distance_l2_vector_u8_batch4`. 4 independent
/// i32 accumulator chains let the OoO core overlap 4 concurrent
/// candidate-stream DRAM misses.
#[inline(never)]
pub fn distance_l2_vector_u8_batch4<const N: usize>(
    a0: &[u8; N],
    a1: &[u8; N],
    a2: &[u8; N],
    a3: &[u8; N],
    q: &[u8; N],
) -> [f32; 4] {
    debug_assert_eq!(N % 16, 0);
    unsafe {
        let mut acc0 = _mm512_setzero_si512();
        let mut acc1 = _mm512_setzero_si512();
        let mut acc2 = _mm512_setzero_si512();
        let mut acc3 = _mm512_setzero_si512();

        let p0 = a0.as_ptr();
        let p1 = a1.as_ptr();
        let p2 = a2.as_ptr();
        let p3 = a3.as_ptr();
        let pq = q.as_ptr();

        let mut i = 0usize;
        while i + 32 <= N {
            let vq = _mm512_cvtepu8_epi16(_mm256_loadu_si256(pq.add(i) as *const __m256i));
            let v0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(p0.add(i) as *const __m256i));
            let v1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(p1.add(i) as *const __m256i));
            let v2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(p2.add(i) as *const __m256i));
            let v3 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(p3.add(i) as *const __m256i));

            let d0 = _mm512_sub_epi16(v0, vq);
            let d1 = _mm512_sub_epi16(v1, vq);
            let d2 = _mm512_sub_epi16(v2, vq);
            let d3 = _mm512_sub_epi16(v3, vq);

            acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(d0, d0));
            acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(d1, d1));
            acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(d2, d2));
            acc3 = _mm512_add_epi32(acc3, _mm512_madd_epi16(d3, d3));
            i += 32;
        }
        // Trailing 16-element chunk (same shape as the single-point path).
        if i + 16 <= N {
            let vq = _mm256_cvtepu8_epi16(_mm_loadu_si128(pq.add(i) as *const __m128i));
            let v0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(p0.add(i) as *const __m128i));
            let v1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(p1.add(i) as *const __m128i));
            let v2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(p2.add(i) as *const __m128i));
            let v3 = _mm256_cvtepu8_epi16(_mm_loadu_si128(p3.add(i) as *const __m128i));

            let d0 = _mm256_sub_epi16(v0, vq);
            let d1 = _mm256_sub_epi16(v1, vq);
            let d2 = _mm256_sub_epi16(v2, vq);
            let d3 = _mm256_sub_epi16(v3, vq);

            acc0 = _mm512_add_epi32(acc0, _mm512_castsi256_si512(_mm256_madd_epi16(d0, d0)));
            acc1 = _mm512_add_epi32(acc1, _mm512_castsi256_si512(_mm256_madd_epi16(d1, d1)));
            acc2 = _mm512_add_epi32(acc2, _mm512_castsi256_si512(_mm256_madd_epi16(d2, d2)));
            acc3 = _mm512_add_epi32(acc3, _mm512_castsi256_si512(_mm256_madd_epi16(d3, d3)));
        }

        [
            reduce_add_epi32(acc0) as f32,
            reduce_add_epi32(acc1) as f32,
            reduce_add_epi32(acc2) as f32,
            reduce_add_epi32(acc3) as f32,
        ]
    }
}
