/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Optional ARM dot-product kernel; all other metrics reuse NEON.
pub use super::neon::IpF32Distance;
pub use super::neon::IpI16Distance;
pub use super::neon::JLHammingDistance;
pub use super::neon::L2F32Distance;
pub use super::neon::L2U16Distance;
pub use super::neon::L2U8Distance;
use crate::DistanceKernelBackend;
use std::arch::aarch64::*;
pub struct IpI8Distance;
impl DistanceKernelBackend for IpI8Distance {
    const MAX_COORDINATES: usize = (i32::MAX as u64 / (128 * 128)) as usize;
    type Storage = i8;
    type Acc = int32x4_t;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { vdupq_n_s32(0) }
    }

    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const i8, query_chunk: *const i8) {
        // FEAT_DotProd `sdot` via inline asm (the
        // instruction sequence is retained from the original kernel).
        // Each `sdot` does 4× i8 multiply-add into i32 (16 i8 lanes
        // per call). 32-byte chunk = 2 sdot ops vs the legacy
        // 4×vmull + 4×vpadalq = 8 ops; dep-chain depth drops 4 → 2
        // on a single accumulator. Native on M1/M2/M3.
        let a_lo = vld1q_s8(base_chunk);
        let a_hi = vld1q_s8(base_chunk.add(16));
        let q_lo = vld1q_s8(query_chunk);
        let q_hi = vld1q_s8(query_chunk.add(16));
        std::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) *acc,
            a   = in(vreg)    a_lo,
            b   = in(vreg)    q_lo,
            options(pure, nomem, nostack),
        );
        std::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) *acc,
            a   = in(vreg)    a_hi,
            b   = in(vreg)    q_hi,
            options(pure, nomem, nostack),
        );
    }

    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { -(vaddvq_s32(acc) as f32) }
    }

    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe { *into = vaddq_s32(*into, src) }
    }
}

pub use super::neon::IpU8Distance;
