/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Private avx512 kernels. Called only through a capability-checked stream entry.

use crate::DistanceKernelBackend;
use std::arch::x86_64::*;
pub struct L2U8Distance;
pub struct L2U16Distance;
pub struct JLHammingDistance;
pub struct IpI8Distance;
pub struct IpI16Distance;
pub struct IpF32Distance;
pub struct L2F32Distance;

impl DistanceKernelBackend for L2U8Distance {
    const MAX_COORDINATES: usize = (u32::MAX as u64 / (255 * 255)) as usize;
    type Storage = u8;
    /// 16 i32 lanes — `_mm512_madd_epi16` produces these directly,
    /// no further widen needed (Σ at D=960 maxes at 62,424,000, within i32).
    type Acc = __m512i;
    const CHUNK_BYTES: usize = 64;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { _mm512_setzero_si512() }
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u8, query_chunk: *const u8) {
        // Split 64 bytes into two 32-byte halves, widen each to
        // 32× i16 via VPMOVZXBW (zero-extend u8 → i16). Difference
        // fits in i16 (range ±255). Squared via VPMADDWD → 16 i32
        // pair sums per half, two halves → 32 i32, summed into acc.
        let a_lo = _mm256_loadu_si256(base_chunk as *const __m256i);
        let a_hi = _mm256_loadu_si256(base_chunk.add(32) as *const __m256i);
        let q_lo = _mm256_loadu_si256(query_chunk as *const __m256i);
        let q_hi = _mm256_loadu_si256(query_chunk.add(32) as *const __m256i);

        let aw_lo = _mm512_cvtepu8_epi16(a_lo);
        let aw_hi = _mm512_cvtepu8_epi16(a_hi);
        let qw_lo = _mm512_cvtepu8_epi16(q_lo);
        let qw_hi = _mm512_cvtepu8_epi16(q_hi);

        let d_lo = _mm512_sub_epi16(aw_lo, qw_lo);
        let d_hi = _mm512_sub_epi16(aw_hi, qw_hi);

        *acc = _mm512_add_epi32(*acc, _mm512_madd_epi16(d_lo, d_lo));
        *acc = _mm512_add_epi32(*acc, _mm512_madd_epi16(d_hi, d_hi));
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { _mm512_reduce_add_epi32(acc) as u32 as f32 }
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe { *into = _mm512_add_epi32(*into, src) }
    }
}

impl DistanceKernelBackend for L2U16Distance {
    const MAX_COORDINATES: usize = (u64::MAX / (65535_u64 * 65535)) as usize;
    type Storage = u16;
    /// Four independent u64 chains, matching the NEON accumulation width.
    type Acc = (__m512i, __m512i, __m512i, __m512i);
    const CHUNK_BYTES: usize = 64;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe {
            (
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            )
        }
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u16, query_chunk: *const u16) {
        // Exactly 64 B = two 256-bit loads of 16 u16 coordinates.
        // Loading 512 bits at offsets 0 and 16 would overlap half a chunk
        // and read beyond the final padded row.
        let a0 = _mm512_cvtepu16_epi32(_mm256_loadu_si256(base_chunk.cast()));
        let a1 = _mm512_cvtepu16_epi32(_mm256_loadu_si256(base_chunk.add(16).cast()));
        let q0 = _mm512_cvtepu16_epi32(_mm256_loadu_si256(query_chunk.cast()));
        let q1 = _mm512_cvtepu16_epi32(_mm256_loadu_si256(query_chunk.add(16).cast()));
        let d0 = _mm512_sub_epi32(a0, q0);
        let d1 = _mm512_sub_epi32(a1, q1);
        let p0 = _mm512_mullo_epi32(d0, d0);
        let p1 = _mm512_mullo_epi32(d1, d1);

        // A squared u16 difference can exceed i32::MAX. Zero-extend the
        // unsigned product into four independent 64-bit accumulator chains.
        acc.0 = _mm512_add_epi64(acc.0, _mm512_cvtepu32_epi64(_mm512_castsi512_si256(p0)));
        acc.1 = _mm512_add_epi64(
            acc.1,
            _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(p0, 1)),
        );
        acc.2 = _mm512_add_epi64(acc.2, _mm512_cvtepu32_epi64(_mm512_castsi512_si256(p1)));
        acc.3 = _mm512_add_epi64(
            acc.3,
            _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(p1, 1)),
        );
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe {
            let s01 = _mm512_add_epi64(acc.0, acc.1);
            let s23 = _mm512_add_epi64(acc.2, acc.3);
            _mm512_reduce_add_epi64(_mm512_add_epi64(s01, s23)) as u64 as f32
        }
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe {
            into.0 = _mm512_add_epi64(into.0, src.0);
            into.1 = _mm512_add_epi64(into.1, src.1);
            into.2 = _mm512_add_epi64(into.2, src.2);
            into.3 = _mm512_add_epi64(into.3, src.3);
        }
    }
}

impl DistanceKernelBackend for JLHammingDistance {
    type Storage = u8;
    /// 8 i64 lanes — VPOPCNTQ output. Per-step max = 64 bytes ×
    /// 8 bits = 512, so even i32 would suffice, but i64 matches
    /// the natural VPOPCNTQ output width with no widening.
    type Acc = __m512i;
    const CHUNK_BYTES: usize = 64;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { _mm512_setzero_si512() }
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u8, query_chunk: *const u8) {
        let a = _mm512_loadu_si512(base_chunk as *const __m512i);
        let q = _mm512_loadu_si512(query_chunk as *const __m512i);
        let x = _mm512_xor_si512(a, q);
        // Byte lookup works on AVX512BW without VPOPCNTDQ. Each 128-bit
        // lane has its own nibble table; SAD sums eight byte counts into u64.
        let lut128 = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
        let lut = _mm512_broadcast_i32x4(lut128);
        let mask = _mm512_set1_epi8(15);
        let lo = _mm512_and_si512(x, mask);
        let hi = _mm512_and_si512(_mm512_srli_epi16(x, 4), mask);
        let counts = _mm512_add_epi8(_mm512_shuffle_epi8(lut, lo), _mm512_shuffle_epi8(lut, hi));
        let pc = _mm512_sad_epu8(counts, _mm512_setzero_si512());
        *acc = _mm512_add_epi64(*acc, pc);
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { _mm512_reduce_add_epi64(acc) as f32 }
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe { *into = _mm512_add_epi64(*into, src) }
    }
}

impl DistanceKernelBackend for IpI8Distance {
    const MAX_COORDINATES: usize = (i32::MAX as u64 / (128 * 128)) as usize;
    type Storage = i8;
    /// 16 i32 lanes — VPMADDWD output. Σ at D=960 maxes ≈ 15.5 M,
    /// well within i32.
    type Acc = __m512i;
    const CHUNK_BYTES: usize = 64;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { _mm512_setzero_si512() }
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const i8, query_chunk: *const i8) {
        // 64 i8 → split into two 32-element halves, sign-extend to
        // i16 via VPMOVSXBW, multiply-add pairs via VPMADDWD.
        let a_lo = _mm256_loadu_si256(base_chunk as *const __m256i);
        let a_hi = _mm256_loadu_si256(base_chunk.add(32) as *const __m256i);
        let q_lo = _mm256_loadu_si256(query_chunk as *const __m256i);
        let q_hi = _mm256_loadu_si256(query_chunk.add(32) as *const __m256i);

        let aw_lo = _mm512_cvtepi8_epi16(a_lo);
        let aw_hi = _mm512_cvtepi8_epi16(a_hi);
        let qw_lo = _mm512_cvtepi8_epi16(q_lo);
        let qw_hi = _mm512_cvtepi8_epi16(q_hi);

        *acc = _mm512_add_epi32(*acc, _mm512_madd_epi16(aw_lo, qw_lo));
        *acc = _mm512_add_epi32(*acc, _mm512_madd_epi16(aw_hi, qw_hi));
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { -(_mm512_reduce_add_epi32(acc) as f32) }
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe { *into = _mm512_add_epi32(*into, src) }
    }
}

impl DistanceKernelBackend for IpI16Distance {
    const MAX_COORDINATES: usize = (i64::MAX as u64 / (32768_u64 * 32768)) as usize;
    type Storage = i16;
    /// Four independent i64 chains, widening each signed product before sum.
    type Acc = (__m512i, __m512i, __m512i, __m512i);
    const CHUNK_BYTES: usize = 64;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe {
            (
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            )
        }
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const i16, query_chunk: *const i16) {
        // Widen before multiplication: madd_epi16 would overflow its
        // signed pair sum for two (-32768 * -32768) products.
        // Two 256-bit loads cover exactly this kernel's 64-byte chunk.
        let a0 = _mm512_cvtepi16_epi32(_mm256_loadu_si256(base_chunk.cast()));
        let a1 = _mm512_cvtepi16_epi32(_mm256_loadu_si256(base_chunk.add(16).cast()));
        let q0 = _mm512_cvtepi16_epi32(_mm256_loadu_si256(query_chunk.cast()));
        let q1 = _mm512_cvtepi16_epi32(_mm256_loadu_si256(query_chunk.add(16).cast()));
        let p0 = _mm512_mullo_epi32(a0, q0);
        let p1 = _mm512_mullo_epi32(a1, q1);
        acc.0 = _mm512_add_epi64(acc.0, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(p0)));
        acc.1 = _mm512_add_epi64(
            acc.1,
            _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(p0, 1)),
        );
        acc.2 = _mm512_add_epi64(acc.2, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(p1)));
        acc.3 = _mm512_add_epi64(
            acc.3,
            _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(p1, 1)),
        );
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe {
            let s01 = _mm512_add_epi64(acc.0, acc.1);
            let s23 = _mm512_add_epi64(acc.2, acc.3);
            -(_mm512_reduce_add_epi64(_mm512_add_epi64(s01, s23)) as f32)
        }
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe {
            into.0 = _mm512_add_epi64(into.0, src.0);
            into.1 = _mm512_add_epi64(into.1, src.1);
            into.2 = _mm512_add_epi64(into.2, src.2);
            into.3 = _mm512_add_epi64(into.3, src.3);
        }
    }
}

impl DistanceKernelBackend for IpF32Distance {
    type Storage = f32;
    /// Single 16-wide FMA accumulator — at 16 floats/chunk the per-
    /// step throughput is already 4 fmadds/cycle on Genoa, well-
    /// saturated by one chain.
    type Acc = __m512;
    const CHUNK_BYTES: usize = 64;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { _mm512_setzero_ps() }
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const f32, query_chunk: *const f32) {
        let a = _mm512_loadu_ps(base_chunk);
        let q = _mm512_loadu_ps(query_chunk);
        *acc = _mm512_fmadd_ps(a, q, *acc);
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { -_mm512_reduce_add_ps(acc) }
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe { *into = _mm512_add_ps(*into, src) }
    }
}

impl DistanceKernelBackend for L2F32Distance {
    type Storage = f32;
    type Acc = __m512;
    const CHUNK_BYTES: usize = 64;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { _mm512_setzero_ps() }
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const f32, query_chunk: *const f32) {
        let a = _mm512_loadu_ps(base_chunk);
        let q = _mm512_loadu_ps(query_chunk);
        let d = _mm512_sub_ps(a, q);
        *acc = _mm512_fmadd_ps(d, d, *acc);
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { _mm512_reduce_add_ps(acc) }
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe { *into = _mm512_add_ps(*into, src) }
    }
}
