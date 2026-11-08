/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! AVX-512 signed-i16 inner product. NEON twin lives at
//! `ip_neon_distance_i16.rs` — see that file for the L2-U16 /
//! high-precision admission rationale and the i64 return contract.
//!
//! AVX-512 pipeline:
//!   * Load 32× i16 per inner iter into a `__m512i`.
//!   * `_mm512_madd_epi16(a, b)` → 16× i32 paired sums.
//!   * Widen to i64 and accumulate (the NEON path keeps i64
//!     accumulators because the per-pair products can overflow
//!     i32 for some dim/scale combinations — we follow the same
//!     conservative widening here).

#![cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]

use std::arch::x86_64::*;

#[inline(always)]
unsafe fn reduce_add_epi64(v: __m512i) -> i64 {
    _mm512_reduce_add_epi64(v)
}

/// Widen a 16-lane i32 vector to two 8-lane i64 vectors and sum.
#[inline(always)]
unsafe fn widen_i32_to_i64_accumulate(acc: __m512i, lanes: __m512i) -> __m512i {
    // Split lanes into low and high 256-bit halves and sign-extend
    // each to 8× i64. Two `_mm512_add_epi64`s fold them into acc.
    let lo = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(lanes));
    let hi = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(lanes, 1));
    let acc = _mm512_add_epi64(acc, lo);
    _mm512_add_epi64(acc, hi)
}

/// Single-pair i16 inner product. Returns `-Σ a_i · b_i` (i64).
#[inline(never)]
pub fn distance_ip_vector_i16<const N: usize>(a: &[i16; N], b: &[i16; N]) -> i64 {
    unsafe {
        let mut acc0 = _mm512_setzero_si512();
        let mut acc1 = _mm512_setzero_si512();
        let mut acc2 = _mm512_setzero_si512();
        let mut acc3 = _mm512_setzero_si512();

        let ap = a.as_ptr();
        let bp = b.as_ptr();

        // 4-way unroll: 128 i16 lanes per outer iter (4 × 32).
        let chunks = N / 128;
        for i in 0..chunks {
            let off = i * 128;

            let va0 = _mm512_loadu_si512(ap.add(off) as *const __m512i);
            let vb0 = _mm512_loadu_si512(bp.add(off) as *const __m512i);
            acc0 = widen_i32_to_i64_accumulate(acc0, _mm512_madd_epi16(va0, vb0));

            let va1 = _mm512_loadu_si512(ap.add(off + 32) as *const __m512i);
            let vb1 = _mm512_loadu_si512(bp.add(off + 32) as *const __m512i);
            acc1 = widen_i32_to_i64_accumulate(acc1, _mm512_madd_epi16(va1, vb1));

            let va2 = _mm512_loadu_si512(ap.add(off + 64) as *const __m512i);
            let vb2 = _mm512_loadu_si512(bp.add(off + 64) as *const __m512i);
            acc2 = widen_i32_to_i64_accumulate(acc2, _mm512_madd_epi16(va2, vb2));

            let va3 = _mm512_loadu_si512(ap.add(off + 96) as *const __m512i);
            let vb3 = _mm512_loadu_si512(bp.add(off + 96) as *const __m512i);
            acc3 = widen_i32_to_i64_accumulate(acc3, _mm512_madd_epi16(va3, vb3));
        }

        // 32-lane tail.
        let mut i = chunks * 128;
        while i + 32 <= N {
            let va = _mm512_loadu_si512(ap.add(i) as *const __m512i);
            let vb = _mm512_loadu_si512(bp.add(i) as *const __m512i);
            acc0 = widen_i32_to_i64_accumulate(acc0, _mm512_madd_epi16(va, vb));
            i += 32;
        }
        // 8-lane chunk (matches NEON path's 8-lane min stride).
        while i + 8 <= N {
            // Load 8 i16 = 16 bytes into the low half of a 256-bit
            // vector. Promote with VPMADDWD at 128-bit width.
            let va = _mm_loadu_si128(ap.add(i) as *const __m128i);
            let vb = _mm_loadu_si128(bp.add(i) as *const __m128i);
            // VPMADDWD → 4× i32. Sign-extend to 4× i64, add to acc0.
            let m = _mm_madd_epi16(va, vb);
            let widened = _mm512_castsi256_si512(_mm256_cvtepi32_epi64(m));
            acc0 = _mm512_add_epi64(acc0, widened);
            i += 8;
        }

        let acc = _mm512_add_epi64(_mm512_add_epi64(acc0, acc1), _mm512_add_epi64(acc2, acc3));
        let mut total = reduce_add_epi64(acc);

        // Scalar tail for the last < 8 elements.
        while i < N {
            total += (a[i] as i64) * (b[i] as i64);
            i += 1;
        }
        -total
    }
}

/// 4 candidates × 1 query batched i16 IP.
#[inline(never)]
pub fn distance_ip_vector_i16_batch4<const N: usize>(
    a0: &[i16; N],
    a1: &[i16; N],
    a2: &[i16; N],
    a3: &[i16; N],
    q: &[i16; N],
) -> [i64; 4] {
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
            let vq = _mm512_loadu_si512(pq.add(i) as *const __m512i);
            let v0 = _mm512_loadu_si512(p0.add(i) as *const __m512i);
            let v1 = _mm512_loadu_si512(p1.add(i) as *const __m512i);
            let v2 = _mm512_loadu_si512(p2.add(i) as *const __m512i);
            let v3 = _mm512_loadu_si512(p3.add(i) as *const __m512i);

            acc0 = widen_i32_to_i64_accumulate(acc0, _mm512_madd_epi16(v0, vq));
            acc1 = widen_i32_to_i64_accumulate(acc1, _mm512_madd_epi16(v1, vq));
            acc2 = widen_i32_to_i64_accumulate(acc2, _mm512_madd_epi16(v2, vq));
            acc3 = widen_i32_to_i64_accumulate(acc3, _mm512_madd_epi16(v3, vq));
            i += 32;
        }
        while i + 8 <= N {
            let vq = _mm_loadu_si128(pq.add(i) as *const __m128i);
            let v0 = _mm_loadu_si128(p0.add(i) as *const __m128i);
            let v1 = _mm_loadu_si128(p1.add(i) as *const __m128i);
            let v2 = _mm_loadu_si128(p2.add(i) as *const __m128i);
            let v3 = _mm_loadu_si128(p3.add(i) as *const __m128i);

            let m0 = _mm512_castsi256_si512(_mm256_cvtepi32_epi64(_mm_madd_epi16(v0, vq)));
            let m1 = _mm512_castsi256_si512(_mm256_cvtepi32_epi64(_mm_madd_epi16(v1, vq)));
            let m2 = _mm512_castsi256_si512(_mm256_cvtepi32_epi64(_mm_madd_epi16(v2, vq)));
            let m3 = _mm512_castsi256_si512(_mm256_cvtepi32_epi64(_mm_madd_epi16(v3, vq)));
            acc0 = _mm512_add_epi64(acc0, m0);
            acc1 = _mm512_add_epi64(acc1, m1);
            acc2 = _mm512_add_epi64(acc2, m2);
            acc3 = _mm512_add_epi64(acc3, m3);
            i += 8;
        }

        let mut r = [
            reduce_add_epi64(acc0),
            reduce_add_epi64(acc1),
            reduce_add_epi64(acc2),
            reduce_add_epi64(acc3),
        ];
        // Scalar tail.
        while i < N {
            let qi = q[i] as i64;
            r[0] += a0[i] as i64 * qi;
            r[1] += a1[i] as i64 * qi;
            r[2] += a2[i] as i64 * qi;
            r[3] += a3[i] as i64 * qi;
            i += 1;
        }
        [-r[0], -r[1], -r[2], -r[3]]
    }
}
