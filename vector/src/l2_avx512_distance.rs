/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! AVX-512 L2 distance kernels for x86_64.
//!
//! 4-way unrolled, 16-wide f32 — 64 floats per outer iter, mirroring
//! the NEON 4-way × 4-wide = 16 floats/iter shape but at AVX-512
//! width. Used for the L2 hot path on Sapphire-Rapids / Genoa-class
//! CPUs.
//!
//! ## Cfg gate
//!
//! Every public function in this file is gated on
//! `cfg(all(target_arch = "x86_64", target_feature = "avx512f"))`.
//! Without that target feature the compiler skips this file
//! entirely; the scalar fallback in `l2_neon_distance.rs` takes
//! over (see `lib.rs` for the routing).

#![cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]

use std::arch::x86_64::*;

use crate::Half;

/// Horizontal sum of a 16-wide f32 register. `_mm512_reduce_add_ps`
/// is an intrinsic-only helper, not a real instruction — the
/// codegen lowers to a butterfly reduction of `vaddps` ops.
#[inline(always)]
unsafe fn reduce_add_ps(v: __m512) -> f32 {
    _mm512_reduce_add_ps(v)
}

/// L2 squared distance, f16 base. AVX-512 FP16 (`avx512fp16`) is
/// Sapphire-Rapids-only and not universally available, so we
/// convert each f16 to f32 element-by-element and run the f32 path.
/// Throughput loss vs a real FP16 kernel is small because the
/// conversion is one BLEND per element and the FMA is the
/// bottleneck either way.
#[inline(never)]
pub fn distance_l2_vector_f16<const N: usize>(a: &[Half; N], b: &[Half; N]) -> f32 {
    let mut a_f32 = [0.0f32; N];
    let mut b_f32 = [0.0f32; N];
    for i in 0..N {
        a_f32[i] = a[i].to_f32();
        b_f32[i] = b[i].to_f32();
    }
    distance_l2_vector_f32(&a_f32, &b_f32)
}

/// L2² between two f32 vectors. 4-way unrolled at 16-wide → 64
/// floats per outer iter. Inputs MUST be 64-byte aligned for best
/// perf (`vmovaps`); we use `vmovups` so misalignment is correct
/// but slightly slower. `AlignedBoxWithSlice` / `AlignedQuery`
/// satisfy 64-byte alignment by default.
#[inline(never)]
pub fn distance_l2_vector_f32<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    debug_assert_eq!(N % 4, 0);

    unsafe {
        let mut sum0 = _mm512_setzero_ps();
        let mut sum1 = _mm512_setzero_ps();
        let mut sum2 = _mm512_setzero_ps();
        let mut sum3 = _mm512_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // 4 outer iters of prefetch lead = 4 × 64 floats = 1024
        // bytes (16 cache lines) — enough to hide L2 latency.
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
            let d0 = _mm512_sub_ps(a0, b0);
            sum0 = _mm512_fmadd_ps(d0, d0, sum0);

            let a1 = _mm512_loadu_ps(a_ptr.add(offset + 16));
            let b1 = _mm512_loadu_ps(b_ptr.add(offset + 16));
            let d1 = _mm512_sub_ps(a1, b1);
            sum1 = _mm512_fmadd_ps(d1, d1, sum1);

            let a2 = _mm512_loadu_ps(a_ptr.add(offset + 32));
            let b2 = _mm512_loadu_ps(b_ptr.add(offset + 32));
            let d2 = _mm512_sub_ps(a2, b2);
            sum2 = _mm512_fmadd_ps(d2, d2, sum2);

            let a3 = _mm512_loadu_ps(a_ptr.add(offset + 48));
            let b3 = _mm512_loadu_ps(b_ptr.add(offset + 48));
            let d3 = _mm512_sub_ps(a3, b3);
            sum3 = _mm512_fmadd_ps(d3, d3, sum3);
        }

        // Tail: process the remaining `N % 64` floats in 16-wide
        // strides plus a final masked tail for the < 16 residue.
        // This covers dim=100 (chunks=1, tail=36) and dim=25
        // (chunks=0, tail=25) cleanly.
        let mut tail_start = chunks * 64;
        while tail_start + 16 <= N {
            let av = _mm512_loadu_ps(a_ptr.add(tail_start));
            let bv = _mm512_loadu_ps(b_ptr.add(tail_start));
            let d = _mm512_sub_ps(av, bv);
            sum0 = _mm512_fmadd_ps(d, d, sum0);
            tail_start += 16;
        }
        let residue = N - tail_start;
        if residue > 0 {
            // `_mm512_maskz_loadu_ps` reads only the elements
            // selected by the mask, zeros the rest. Same trick
            // works for the diff — zero-extended dims contribute
            // 0² = 0 to the sum.
            let mask: __mmask16 = ((1u32 << residue) - 1) as __mmask16;
            let av = _mm512_maskz_loadu_ps(mask, a_ptr.add(tail_start));
            let bv = _mm512_maskz_loadu_ps(mask, b_ptr.add(tail_start));
            let d = _mm512_sub_ps(av, bv);
            sum0 = _mm512_fmadd_ps(d, d, sum0);
        }

        let total = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
        reduce_add_ps(total)
    }
}

/// ADSampling-style L2 distance with probabilistic early abandon.
/// See `l2_neon_distance::distance_l2_adsampling_f32` for the math.
/// Check interval is 1 outer iter (= 64 dims), matching NEON's
/// 4-outer-iter cadence of 64 dims.
#[inline(never)]
pub fn distance_l2_adsampling_f32<const N: usize>(
    a: &[f32; N],
    b: &[f32; N],
    upper_bound: f32,
    epsilon: f32,
) -> f32 {
    debug_assert_eq!(N % 4, 0);

    unsafe {
        let mut sum0 = _mm512_setzero_ps();
        let mut sum1 = _mm512_setzero_ps();
        let mut sum2 = _mm512_setzero_ps();
        let mut sum3 = _mm512_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        const PF_AHEAD: usize = 4;
        let chunks = N / 64;
        let n_f = N as f32;

        for i in 0..chunks {
            let offset = i * 64;

            if i + PF_AHEAD < chunks {
                let pf_offset = (i + PF_AHEAD) * 64;
                _mm_prefetch(a_ptr.add(pf_offset) as *const i8, _MM_HINT_T0);
                _mm_prefetch(b_ptr.add(pf_offset) as *const i8, _MM_HINT_T0);
            }

            let a0 = _mm512_loadu_ps(a_ptr.add(offset));
            let b0 = _mm512_loadu_ps(b_ptr.add(offset));
            let d0 = _mm512_sub_ps(a0, b0);
            sum0 = _mm512_fmadd_ps(d0, d0, sum0);

            let a1 = _mm512_loadu_ps(a_ptr.add(offset + 16));
            let b1 = _mm512_loadu_ps(b_ptr.add(offset + 16));
            let d1 = _mm512_sub_ps(a1, b1);
            sum1 = _mm512_fmadd_ps(d1, d1, sum1);

            let a2 = _mm512_loadu_ps(a_ptr.add(offset + 32));
            let b2 = _mm512_loadu_ps(b_ptr.add(offset + 32));
            let d2 = _mm512_sub_ps(a2, b2);
            sum2 = _mm512_fmadd_ps(d2, d2, sum2);

            let a3 = _mm512_loadu_ps(a_ptr.add(offset + 48));
            let b3 = _mm512_loadu_ps(b_ptr.add(offset + 48));
            let d3 = _mm512_sub_ps(a3, b3);
            sum3 = _mm512_fmadd_ps(d3, d3, sum3);

            // Check every outer iter (= 64 dims processed so far).
            // The reduce-add is one butterfly; horizontal-reduce
            // latency is hidden by FMA throughput at this cadence.
            let d_prime = ((i + 1) * 64) as f32;
            let partial_total = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
            let partial = reduce_add_ps(partial_total);
            let scaled = partial * (n_f / d_prime);
            let margin = upper_bound * (1.0 + epsilon / d_prime.sqrt());
            if scaled > margin {
                return -1.0;
            }
        }

        // Tail (same shape as `distance_l2_vector_f32`).
        let mut tail_start = chunks * 64;
        while tail_start + 16 <= N {
            let av = _mm512_loadu_ps(a_ptr.add(tail_start));
            let bv = _mm512_loadu_ps(b_ptr.add(tail_start));
            let d = _mm512_sub_ps(av, bv);
            sum0 = _mm512_fmadd_ps(d, d, sum0);
            tail_start += 16;
        }
        let residue = N - tail_start;
        if residue > 0 {
            let mask: __mmask16 = ((1u32 << residue) - 1) as __mmask16;
            let av = _mm512_maskz_loadu_ps(mask, a_ptr.add(tail_start));
            let bv = _mm512_maskz_loadu_ps(mask, b_ptr.add(tail_start));
            let d = _mm512_sub_ps(av, bv);
            sum0 = _mm512_fmadd_ps(d, d, sum0);
        }

        let total = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
        reduce_add_ps(total)
    }
}

/// L2² with a hard early-abandon — return `-1.0` if any partial
/// sum exceeds `upper_bound`. Used by the navigation tier when the
/// caller has a tight cutoff (e.g. PA-aligned merge-then-rerank).
#[inline(never)]
pub fn distance_l2_early_abandon_f32<const N: usize>(
    a: &[f32; N],
    b: &[f32; N],
    upper_bound: f32,
) -> f32 {
    debug_assert_eq!(N % 4, 0);

    unsafe {
        let mut sum0 = _mm512_setzero_ps();
        let mut sum1 = _mm512_setzero_ps();
        let mut sum2 = _mm512_setzero_ps();
        let mut sum3 = _mm512_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        const PF_AHEAD: usize = 4;
        // Same 128-dim check cadence as the NEON path (NEON had
        // CHECK_INTERVAL=8 over 16-dim outer iters; we have
        // 64-dim outer iters so 128-dim cadence is every 2 iters).
        const CHECK_INTERVAL: usize = 2;

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
            let d0 = _mm512_sub_ps(a0, b0);
            sum0 = _mm512_fmadd_ps(d0, d0, sum0);

            let a1 = _mm512_loadu_ps(a_ptr.add(offset + 16));
            let b1 = _mm512_loadu_ps(b_ptr.add(offset + 16));
            let d1 = _mm512_sub_ps(a1, b1);
            sum1 = _mm512_fmadd_ps(d1, d1, sum1);

            let a2 = _mm512_loadu_ps(a_ptr.add(offset + 32));
            let b2 = _mm512_loadu_ps(b_ptr.add(offset + 32));
            let d2 = _mm512_sub_ps(a2, b2);
            sum2 = _mm512_fmadd_ps(d2, d2, sum2);

            let a3 = _mm512_loadu_ps(a_ptr.add(offset + 48));
            let b3 = _mm512_loadu_ps(b_ptr.add(offset + 48));
            let d3 = _mm512_sub_ps(a3, b3);
            sum3 = _mm512_fmadd_ps(d3, d3, sum3);

            if (i + 1) % CHECK_INTERVAL == 0 {
                let partial_total =
                    _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
                if reduce_add_ps(partial_total) >= upper_bound {
                    return -1.0;
                }
            }
        }

        // Tail.
        let mut tail_start = chunks * 64;
        while tail_start + 16 <= N {
            let av = _mm512_loadu_ps(a_ptr.add(tail_start));
            let bv = _mm512_loadu_ps(b_ptr.add(tail_start));
            let d = _mm512_sub_ps(av, bv);
            sum0 = _mm512_fmadd_ps(d, d, sum0);
            tail_start += 16;
        }
        let residue = N - tail_start;
        if residue > 0 {
            let mask: __mmask16 = ((1u32 << residue) - 1) as __mmask16;
            let av = _mm512_maskz_loadu_ps(mask, a_ptr.add(tail_start));
            let bv = _mm512_maskz_loadu_ps(mask, b_ptr.add(tail_start));
            let d = _mm512_sub_ps(av, bv);
            sum0 = _mm512_fmadd_ps(d, d, sum0);
        }

        let total = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
        let dist = reduce_add_ps(total);
        if dist >= upper_bound {
            -1.0
        } else {
            dist
        }
    }
}
