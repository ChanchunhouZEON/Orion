/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! VPOPCNT-only override; entered after detecting the extra CPU feature.
pub use super::avx512::IpF32Distance;
pub use super::avx512::IpI16Distance;
pub use super::avx512::IpI8Distance;
pub use super::avx512::L2F32Distance;
pub use super::avx512::L2U16Distance;
pub use super::avx512::L2U8Distance;
use crate::DistanceKernelBackend;
use std::arch::x86_64::*;
pub struct JLHammingDistance;
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
        // Entry requires VPOPCNTDQ in addition to the base AVX512 features.
        let pc = _mm512_popcnt_epi64(x);
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

pub use super::avx512::IpU8Distance;
