//! Private neon kernels. Called only through a capability-checked stream entry.

use crate::DistanceKernelBackend;
use std::arch::aarch64::*;
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
    type Acc = uint32x4_t;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { vdupq_n_u32(0) }
    }

    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u8, query_chunk: *const u8) {
        let a_lo = vld1q_u8(base_chunk);
        let a_hi = vld1q_u8(base_chunk.add(16));
        let q_lo = vld1q_u8(query_chunk);
        let q_hi = vld1q_u8(query_chunk.add(16));

        let d_lo = vabdq_u8(a_lo, q_lo);
        let d_hi = vabdq_u8(a_hi, q_hi);

        // Squared diffs widened to u16x8, pairwise-summed into the
        // running u32x4 accumulator. No horizontal reduce here.
        let p_lo_l = vmull_u8(vget_low_u8(d_lo), vget_low_u8(d_lo));
        let p_lo_h = vmull_high_u8(d_lo, d_lo);
        let p_hi_l = vmull_u8(vget_low_u8(d_hi), vget_low_u8(d_hi));
        let p_hi_h = vmull_high_u8(d_hi, d_hi);

        *acc = vpadalq_u16(*acc, p_lo_l);
        *acc = vpadalq_u16(*acc, p_lo_h);
        *acc = vpadalq_u16(*acc, p_hi_l);
        *acc = vpadalq_u16(*acc, p_hi_h);
    }

    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { vaddvq_u32(acc) as f32 }
    }

    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe { *into = vaddq_u32(*into, src) }
    }
}

impl DistanceKernelBackend for L2U16Distance {
    const MAX_COORDINATES: usize = (u64::MAX / (65535_u64 * 65535)) as usize;
    type Storage = u16;
    type Acc = (uint64x2_t, uint64x2_t, uint64x2_t, uint64x2_t);

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe {
            (
                vdupq_n_u64(0),
                vdupq_n_u64(0),
                vdupq_n_u64(0),
                vdupq_n_u64(0),
            )
        }
    }

    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u16, query_chunk: *const u16) {
        // 32 bytes = 16 u16 lanes = two paired loads of u16x8.
        let a_lo = vld1q_u16(base_chunk);
        let a_hi = vld1q_u16(base_chunk.add(8));
        let q_lo = vld1q_u16(query_chunk);
        let q_hi = vld1q_u16(query_chunk.add(8));

        // `|a - b|` as u16 — `vabdq_u16` is one instruction.
        let d_lo = vabdq_u16(a_lo, q_lo);
        let d_hi = vabdq_u16(a_hi, q_hi);

        // Squared diffs: `vmull_u16` widens u16x4 → u32x4. Four u32x4
        // vectors per step (one per low/high half of d_lo and d_hi).
        let p_ll = vmull_u16(vget_low_u16(d_lo), vget_low_u16(d_lo));
        let p_lh = vmull_high_u16(d_lo, d_lo);
        let p_hl = vmull_u16(vget_low_u16(d_hi), vget_low_u16(d_hi));
        let p_hh = vmull_high_u16(d_hi, d_hi);

        // Pairwise-add widen u32x4 → u64x2 into 4 independent chains.
        // Splitting across chains breaks the `vpadalq_u32` dep chain so
        // the M2 OoO engine can issue all four in parallel.
        acc.0 = vpadalq_u32(acc.0, p_ll);
        acc.1 = vpadalq_u32(acc.1, p_lh);
        acc.2 = vpadalq_u32(acc.2, p_hl);
        acc.3 = vpadalq_u32(acc.3, p_hh);
    }

    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe {
            let s = vaddq_u64(vaddq_u64(acc.0, acc.1), vaddq_u64(acc.2, acc.3));
            // u64 → f32: at D=960 / u16 max range the value can reach
            // 4 TB which exceeds f32 mantissa (16.7 M). Distance
            // ordering is preserved by the monotone cast, so PQ
            // comparisons stay correct even at precision-loss scale.
            vaddvq_u64(s) as f32
        }
    }

    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe {
            into.0 = vaddq_u64(into.0, src.0);
            into.1 = vaddq_u64(into.1, src.1);
            into.2 = vaddq_u64(into.2, src.2);
            into.3 = vaddq_u64(into.3, src.3);
        }
    }
}

impl DistanceKernelBackend for JLHammingDistance {
    type Storage = u8;
    type Acc = uint32x4_t;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { vdupq_n_u32(0) }
    }

    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u8, query_chunk: *const u8) {
        let a_lo = vld1q_u8(base_chunk);
        let a_hi = vld1q_u8(base_chunk.add(16));
        let q_lo = vld1q_u8(query_chunk);
        let q_hi = vld1q_u8(query_chunk.add(16));

        // Per-byte popcount of XOR — `vcntq_u8` is one instruction that
        // returns 16 bytes whose lanes each hold the popcount (0..=8)
        // of the corresponding input byte.
        let p_lo = vcntq_u8(veorq_u8(a_lo, q_lo));
        let p_hi = vcntq_u8(veorq_u8(a_hi, q_hi));

        // Pairwise-add widen u8 → u16 → u32, fold into running u32x4
        // acc. Each `vpaddlq_u8` pair-sums 16 u8 lanes into 8 u16
        // lanes; `vpaddlq_u16` then pair-sums 8 u16 lanes into 4 u32
        // lanes. The two halves accumulate independently (no inter-
        // half dep), keeping the M2 NEON pipes full.
        let h16_lo = vpaddlq_u8(p_lo);
        let h16_hi = vpaddlq_u8(p_hi);
        let h32_lo = vpaddlq_u16(h16_lo);
        let h32_hi = vpaddlq_u16(h16_hi);

        *acc = vaddq_u32(*acc, h32_lo);
        *acc = vaddq_u32(*acc, h32_hi);
    }

    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { vaddvq_u32(acc) as f32 }
    }

    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe { *into = vaddq_u32(*into, src) }
    }
}

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
        let a0 = vld1q_s8(base_chunk);
        let a1 = vld1q_s8(base_chunk.add(16));
        let q0 = vld1q_s8(query_chunk);
        let q1 = vld1q_s8(query_chunk.add(16));
        *acc = vpadalq_s16(*acc, vmull_s8(vget_low_s8(a0), vget_low_s8(q0)));
        *acc = vpadalq_s16(*acc, vmull_high_s8(a0, q0));
        *acc = vpadalq_s16(*acc, vmull_s8(vget_low_s8(a1), vget_low_s8(q1)));
        *acc = vpadalq_s16(*acc, vmull_high_s8(a1, q1));
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

impl DistanceKernelBackend for IpI16Distance {
    const MAX_COORDINATES: usize = (i64::MAX as u64 / (32768_u64 * 32768)) as usize;
    type Storage = i16;
    /// 4 independent `int64x2_t` accumulators — same shape PA's
    /// well-tuned i16 kernel uses. Single-acc was the bottleneck:
    /// `vpadalq_s32` has ~4-cycle latency, four serial calls per
    /// chunk = 16 cycle critical path. With 4 acc chains the four
    /// `vpadalq_s32` calls in one step run in parallel through M2's
    /// 4 NEON pipes, dropping per-chunk latency to ~4 cycle.
    type Acc = (int64x2_t, int64x2_t, int64x2_t, int64x2_t);

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe {
            (
                vdupq_n_s64(0),
                vdupq_n_s64(0),
                vdupq_n_s64(0),
                vdupq_n_s64(0),
            )
        }
    }

    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const i16, query_chunk: *const i16) {
        // 32 bytes = 16 i16 lanes = two paired loads of i16x8.
        let a_lo = vld1q_s16(base_chunk);
        let a_hi = vld1q_s16(base_chunk.add(8));
        let q_lo = vld1q_s16(query_chunk);
        let q_hi = vld1q_s16(query_chunk.add(8));

        // vmull_s16 widens 4 i16 lanes to 4 i32 products. Each goes
        // into its own independent acc chain — no serial dependency.
        let p_lo_l = vmull_s16(vget_low_s16(a_lo), vget_low_s16(q_lo));
        let p_lo_h = vmull_high_s16(a_lo, q_lo);
        let p_hi_l = vmull_s16(vget_low_s16(a_hi), vget_low_s16(q_hi));
        let p_hi_h = vmull_high_s16(a_hi, q_hi);

        acc.0 = vpadalq_s32(acc.0, p_lo_l);
        acc.1 = vpadalq_s32(acc.1, p_lo_h);
        acc.2 = vpadalq_s32(acc.2, p_hi_l);
        acc.3 = vpadalq_s32(acc.3, p_hi_h);
    }

    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        // Tree reduce: 3 vaddq + 1 vaddvq + sign flip.
        unsafe {
            let s01 = vaddq_s64(acc.0, acc.1);
            let s23 = vaddq_s64(acc.2, acc.3);
            -(vaddvq_s64(vaddq_s64(s01, s23)) as f32)
        }
    }

    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe {
            into.0 = vaddq_s64(into.0, src.0);
            into.1 = vaddq_s64(into.1, src.1);
            into.2 = vaddq_s64(into.2, src.2);
            into.3 = vaddq_s64(into.3, src.3);
        }
    }
}

impl DistanceKernelBackend for IpF32Distance {
    type Storage = f32;
    type Acc = (float32x4_t, float32x4_t);

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { (vdupq_n_f32(0.0), vdupq_n_f32(0.0)) }
    }

    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const f32, query_chunk: *const f32) {
        let a_lo = vld1q_f32(base_chunk);
        let a_hi = vld1q_f32(base_chunk.add(4));
        let q_lo = vld1q_f32(query_chunk);
        let q_hi = vld1q_f32(query_chunk.add(4));
        acc.0 = vfmaq_f32(acc.0, a_lo, q_lo);
        acc.1 = vfmaq_f32(acc.1, a_hi, q_hi);
    }

    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { -vaddvq_f32(vaddq_f32(acc.0, acc.1)) }
    }

    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe {
            into.0 = vaddq_f32(into.0, src.0);
            into.1 = vaddq_f32(into.1, src.1);
        }
    }
}

impl DistanceKernelBackend for L2F32Distance {
    type Storage = f32;
    type Acc = (float32x4_t, float32x4_t);

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { (vdupq_n_f32(0.0), vdupq_n_f32(0.0)) }
    }

    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const f32, query_chunk: *const f32) {
        let a_lo = vld1q_f32(base_chunk);
        let a_hi = vld1q_f32(base_chunk.add(4));
        let q_lo = vld1q_f32(query_chunk);
        let q_hi = vld1q_f32(query_chunk.add(4));
        let d_lo = vsubq_f32(a_lo, q_lo);
        let d_hi = vsubq_f32(a_hi, q_hi);
        acc.0 = vfmaq_f32(acc.0, d_lo, d_lo);
        acc.1 = vfmaq_f32(acc.1, d_hi, d_hi);
    }

    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { vaddvq_f32(vaddq_f32(acc.0, acc.1)) }
    }

    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe {
            into.0 = vaddq_f32(into.0, src.0);
            into.1 = vaddq_f32(into.1, src.1);
        }
    }
}
