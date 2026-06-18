/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! AVX-512 negative-inner-product distance for x86_64.
//!
//! See `ip_neon_distance.rs` for the NEON version + the
//! `-⟨a,b⟩` ranking contract used by the rest of the search
//! code. AVX-512 doubles NEON's per-iter throughput by going
//! 16-wide, so 4-way unrolling here lands at 64 floats per
//! outer iter (vs NEON's 16).

#![cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]

use std::arch::x86_64::*;

#[inline(always)]
unsafe fn reduce_add_ps(v: __m512) -> f32 {
    _mm512_reduce_add_ps(v)
}

/// Returns `-⟨a, b⟩` (smaller == closer). 4-way unrolled at
/// 16-wide. See module docstring for unroll factor rationale.
#[inline(never)]
pub fn distance_ip_vector_f32<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    debug_assert_eq!(N % 4, 0);
    unsafe {
        let mut s0 = _mm512_setzero_ps();
        let mut s1 = _mm512_setzero_ps();
        let mut s2 = _mm512_setzero_ps();
        let mut s3 = _mm512_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        const PF_AHEAD: usize = 4;
        let chunks = N / 64;
        for i in 0..chunks {
            let offset = i * 64;
            if i + PF_AHEAD < chunks {
                let pf_offset = (i + PF_AHEAD) * 64;
                _mm_prefetch(a_ptr.add(pf_offset) as *const i8, _MM_HINT_T0);
                _mm_prefetch(b_ptr.add(pf_offset) as *const i8, _MM_HINT_T0);
            }
            let a0 = _mm512_loadu_ps(a_ptr.add(offset));
            let b0 = _mm512_loadu_ps(b_ptr.add(offset));
            s0 = _mm512_fmadd_ps(a0, b0, s0);

            let a1 = _mm512_loadu_ps(a_ptr.add(offset + 16));
            let b1 = _mm512_loadu_ps(b_ptr.add(offset + 16));
            s1 = _mm512_fmadd_ps(a1, b1, s1);

            let a2 = _mm512_loadu_ps(a_ptr.add(offset + 32));
            let b2 = _mm512_loadu_ps(b_ptr.add(offset + 32));
            s2 = _mm512_fmadd_ps(a2, b2, s2);

            let a3 = _mm512_loadu_ps(a_ptr.add(offset + 48));
            let b3 = _mm512_loadu_ps(b_ptr.add(offset + 48));
            s3 = _mm512_fmadd_ps(a3, b3, s3);
        }

        let mut tail = chunks * 64;
        while tail + 16 <= N {
            let av = _mm512_loadu_ps(a_ptr.add(tail));
            let bv = _mm512_loadu_ps(b_ptr.add(tail));
            s0 = _mm512_fmadd_ps(av, bv, s0);
            tail += 16;
        }
        let residue = N - tail;
        if residue > 0 {
            let mask: __mmask16 = ((1u32 << residue) - 1) as __mmask16;
            let av = _mm512_maskz_loadu_ps(mask, a_ptr.add(tail));
            let bv = _mm512_maskz_loadu_ps(mask, b_ptr.add(tail));
            s0 = _mm512_fmadd_ps(av, bv, s0);
        }

        let total = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
        -reduce_add_ps(total)
    }
}

/// 4-candidate × 1-query batched f32 IP. Same `-Σ(a_k · q)`
/// contract as `distance_ip_vector_f32`, four results in one call.
/// 4 independent accumulator chains overlap 4 concurrent DRAM
/// misses through the MSHR queue — see the NEON docstring for the
/// MIPS-Q cold-cache rationale.
#[inline(never)]
pub fn distance_ip_vector_f32_batch4<const N: usize>(
    a0: &[f32; N],
    a1: &[f32; N],
    a2: &[f32; N],
    a3: &[f32; N],
    q: &[f32; N],
) -> [f32; 4] {
    debug_assert_eq!(N % 4, 0);
    unsafe {
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        let p0 = a0.as_ptr();
        let p1 = a1.as_ptr();
        let p2 = a2.as_ptr();
        let p3 = a3.as_ptr();
        let pq = q.as_ptr();

        let mut i = 0usize;
        while i + 16 <= N {
            let vq = _mm512_loadu_ps(pq.add(i));
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(p0.add(i)), vq, acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(p1.add(i)), vq, acc1);
            acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(p2.add(i)), vq, acc2);
            acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(p3.add(i)), vq, acc3);
            i += 16;
        }
        if i < N {
            let residue = N - i;
            let mask: __mmask16 = ((1u32 << residue) - 1) as __mmask16;
            let vq = _mm512_maskz_loadu_ps(mask, pq.add(i));
            acc0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, p0.add(i)), vq, acc0);
            acc1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, p1.add(i)), vq, acc1);
            acc2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, p2.add(i)), vq, acc2);
            acc3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, p3.add(i)), vq, acc3);
        }

        [
            -reduce_add_ps(acc0),
            -reduce_add_ps(acc1),
            -reduce_add_ps(acc2),
            -reduce_add_ps(acc3),
        ]
    }
}
