/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! AVX-512 signed-i8 inner product. NEON twin lives at
//! `ip_neon_distance_i8.rs`; see that file for the MIPS-Q /
//! angular-search rationale and the "smaller == closer" contract.
//!
//! AVX-512 pipeline (no VNNI required, works on every AVX-512BW
//! CPU back to Skylake-SP):
//!   * Load 32 i8 bytes into `__m256i`.
//!   * Sign-extend to 32× i16 via `_mm512_cvtepi8_epi16`.
//!   * `_mm512_madd_epi16(a, b)` → 16× i32 paired sums.
//!   * Accumulate into 512-bit i32 register.
//!
//! With VNNI (`avx512vnni`, Ice Lake+) a single
//! `_mm512_dpbusd_epi32` could replace cvt+sub+madd at i8 width,
//! but we stay on the BW path for max CPU coverage and let the
//! compiler peep-hole the VNNI rewrite if the user passes
//! `target-feature=+avx512vnni`.

#![cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]

use std::arch::x86_64::*;

#[inline(always)]
unsafe fn reduce_add_epi32(v: __m512i) -> i32 {
    _mm512_reduce_add_epi32(v)
}

/// Single-pair i8 inner product. Returns `-Σ a_i · b_i` (i32).
#[inline(never)]
pub fn distance_ip_vector_i8<const N: usize>(a: &[i8; N], b: &[i8; N]) -> i32 {
    unsafe {
        let mut acc0 = _mm512_setzero_si512();
        let mut acc1 = _mm512_setzero_si512();
        let mut acc2 = _mm512_setzero_si512();
        let mut acc3 = _mm512_setzero_si512();

        let ap = a.as_ptr();
        let bp = b.as_ptr();

        // 4-way unroll: 128 i8 lanes per outer iter (4 × 32).
        let chunks = N / 128;
        for i in 0..chunks {
            let off = i * 128;

            let va0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(ap.add(off) as *const __m256i));
            let vb0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(bp.add(off) as *const __m256i));
            acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(va0, vb0));

            let va1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(ap.add(off + 32) as *const __m256i));
            let vb1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(bp.add(off + 32) as *const __m256i));
            acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(va1, vb1));

            let va2 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(ap.add(off + 64) as *const __m256i));
            let vb2 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(bp.add(off + 64) as *const __m256i));
            acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(va2, vb2));

            let va3 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(ap.add(off + 96) as *const __m256i));
            let vb3 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(bp.add(off + 96) as *const __m256i));
            acc3 = _mm512_add_epi32(acc3, _mm512_madd_epi16(va3, vb3));
        }

        // Tail in 32-lane strides.
        let mut i = chunks * 128;
        while i + 32 <= N {
            let va = _mm512_cvtepi8_epi16(_mm256_loadu_si256(ap.add(i) as *const __m256i));
            let vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256(bp.add(i) as *const __m256i));
            acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(va, vb));
            i += 32;
        }
        // 16-lane chunk.
        if i + 16 <= N {
            let va = _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i) as *const __m128i));
            let vb = _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i) as *const __m128i));
            acc0 = _mm512_add_epi32(acc0, _mm512_castsi256_si512(_mm256_madd_epi16(va, vb)));
            i += 16;
        }

        let acc = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1), _mm512_add_epi32(acc2, acc3));
        let mut total = reduce_add_epi32(acc);

        // Scalar tail for the last < 16 elements.
        while i < N {
            total += (a[i] as i32) * (b[i] as i32);
            i += 1;
        }
        -total
    }
}

/// 4 candidates × 1 query batched i8 IP.
#[inline(never)]
pub fn distance_ip_vector_i8_batch4<const N: usize>(
    a0: &[i8; N],
    a1: &[i8; N],
    a2: &[i8; N],
    a3: &[i8; N],
    q: &[i8; N],
) -> [i32; 4] {
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
            let vq = _mm512_cvtepi8_epi16(_mm256_loadu_si256(pq.add(i) as *const __m256i));
            let v0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(p0.add(i) as *const __m256i));
            let v1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(p1.add(i) as *const __m256i));
            let v2 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(p2.add(i) as *const __m256i));
            let v3 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(p3.add(i) as *const __m256i));

            acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(v0, vq));
            acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(v1, vq));
            acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(v2, vq));
            acc3 = _mm512_add_epi32(acc3, _mm512_madd_epi16(v3, vq));
            i += 32;
        }
        if i + 16 <= N {
            let vq = _mm256_cvtepi8_epi16(_mm_loadu_si128(pq.add(i) as *const __m128i));
            let v0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(p0.add(i) as *const __m128i));
            let v1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(p1.add(i) as *const __m128i));
            let v2 = _mm256_cvtepi8_epi16(_mm_loadu_si128(p2.add(i) as *const __m128i));
            let v3 = _mm256_cvtepi8_epi16(_mm_loadu_si128(p3.add(i) as *const __m128i));

            acc0 = _mm512_add_epi32(acc0, _mm512_castsi256_si512(_mm256_madd_epi16(v0, vq)));
            acc1 = _mm512_add_epi32(acc1, _mm512_castsi256_si512(_mm256_madd_epi16(v1, vq)));
            acc2 = _mm512_add_epi32(acc2, _mm512_castsi256_si512(_mm256_madd_epi16(v2, vq)));
            acc3 = _mm512_add_epi32(acc3, _mm512_castsi256_si512(_mm256_madd_epi16(v3, vq)));
            i += 16;
        }

        let mut r = [
            reduce_add_epi32(acc0),
            reduce_add_epi32(acc1),
            reduce_add_epi32(acc2),
            reduce_add_epi32(acc3),
        ];
        // Scalar tail.
        while i < N {
            let qi = q[i] as i32;
            r[0] += a0[i] as i32 * qi;
            r[1] += a1[i] as i32 * qi;
            r[2] += a2[i] as i32 * qi;
            r[3] += a3[i] as i32 * qi;
            i += 1;
        }
        [-r[0], -r[1], -r[2], -r[3]]
    }
}
