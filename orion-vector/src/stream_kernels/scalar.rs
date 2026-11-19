/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Private scalar kernels. Called only through a capability-checked stream entry.

use crate::DistanceKernelBackend;

/// Private scalar implementation of the squared-L2 byte stream kernel.
///
/// Use the public distance facade so CPU capability checks precede execution.
pub struct L2U8Distance;

/// Private scalar implementation of the squared-L2 unsigned-16-bit stream kernel.
///
/// Use the public distance facade so CPU capability checks precede execution.
pub struct L2U16Distance;

/// Private scalar implementation of the binary Hamming stream kernel.
///
/// Use the public distance facade so CPU capability checks precede execution.
pub struct JLHammingDistance;

/// Private scalar implementation of the signed-byte inner-product stream kernel.
///
/// Use the public distance facade so CPU capability checks precede execution.
pub struct IpI8Distance;

/// Private scalar implementation of the signed-16-bit inner-product stream kernel.
///
/// Use the public distance facade so CPU capability checks precede execution.
pub struct IpI16Distance;

/// Private scalar implementation of the f32 inner-product stream kernel.
///
/// Use the public distance facade so CPU capability checks precede execution.
pub struct IpF32Distance;

/// Private scalar implementation of the f32 squared-L2 stream kernel.
///
/// Use the public distance facade so CPU capability checks precede execution.
pub struct L2F32Distance;

impl DistanceKernelBackend for L2U8Distance {
    const MAX_COORDINATES: usize = (u32::MAX as u64 / (255 * 255)) as usize;
    type Storage = u8;
    type Acc = u32;

    /// Starts a fresh per-vector accumulator for this distance kernel.
    ///
    /// See [`DistanceKernelBackend::init`] for the shared contract.
    #[inline(always)]
    fn init() -> Self::Acc {
        0
    }

    /// Accumulates one kernel chunk without reducing the partial distance.
    ///
    /// # Safety
    ///
    /// The caller must uphold [`DistanceKernelBackend::step`]'s pointer, shape and lifetime contract.
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u8, query_chunk: *const u8) {
        for i in 0..32 {
            let a = *base_chunk.add(i) as i32;
            let b = *query_chunk.add(i) as i32;
            // Keep the signed difference until after squaring: casting a
            // negative value to u32 first overflows the debug-mode multiply.
            // u8 differences are in [-255, 255], so the i32 square is safe.
            let d = a - b;
            *acc = acc.wrapping_add((d * d) as u32);
        }
    }

    /// Converts the completed accumulator to a smaller-is-closer distance.
    ///
    /// See [`DistanceKernelBackend::reduce`] for the shared contract.
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        acc as f32
    }

    /// Combines independent chunk accumulators before the final reduction.
    ///
    /// See [`DistanceKernelBackend::merge`] for the shared contract.
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into = into.wrapping_add(src);
    }
}

impl DistanceKernelBackend for L2U16Distance {
    const MAX_COORDINATES: usize = (u64::MAX / (65535_u64 * 65535)) as usize;
    type Storage = u16;
    type Acc = u64;

    /// Starts a fresh per-vector accumulator for this distance kernel.
    ///
    /// See [`DistanceKernelBackend::init`] for the shared contract.
    #[inline(always)]
    fn init() -> Self::Acc {
        0
    }

    /// Accumulates one kernel chunk without reducing the partial distance.
    ///
    /// # Safety
    ///
    /// The caller must uphold [`DistanceKernelBackend::step`]'s pointer, shape and lifetime contract.
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u16, query_chunk: *const u16) {
        for i in 0..16 {
            let a = *base_chunk.add(i) as i32;
            let b = *query_chunk.add(i) as i32;
            let d = (a - b).unsigned_abs() as u64;
            *acc = acc.wrapping_add(d * d);
        }
    }

    /// Converts the completed accumulator to a smaller-is-closer distance.
    ///
    /// See [`DistanceKernelBackend::reduce`] for the shared contract.
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        acc as f32
    }

    /// Combines independent chunk accumulators before the final reduction.
    ///
    /// See [`DistanceKernelBackend::merge`] for the shared contract.
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into = into.wrapping_add(src);
    }
}

impl DistanceKernelBackend for JLHammingDistance {
    type Storage = u8;
    type Acc = u32;

    /// Starts a fresh per-vector accumulator for this distance kernel.
    ///
    /// See [`DistanceKernelBackend::init`] for the shared contract.
    #[inline(always)]
    fn init() -> Self::Acc {
        0
    }

    /// Accumulates one kernel chunk without reducing the partial distance.
    ///
    /// # Safety
    ///
    /// The caller must uphold [`DistanceKernelBackend::step`]'s pointer, shape and lifetime contract.
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u8, query_chunk: *const u8) {
        for i in 0..32 {
            *acc = acc.wrapping_add(((*base_chunk.add(i)) ^ (*query_chunk.add(i))).count_ones());
        }
    }

    /// Converts the completed accumulator to a smaller-is-closer distance.
    ///
    /// See [`DistanceKernelBackend::reduce`] for the shared contract.
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        acc as f32
    }

    /// Combines independent chunk accumulators before the final reduction.
    ///
    /// See [`DistanceKernelBackend::merge`] for the shared contract.
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into = into.wrapping_add(src);
    }
}

impl DistanceKernelBackend for IpI8Distance {
    const MAX_COORDINATES: usize = (i32::MAX as u64 / (128 * 128)) as usize;
    type Storage = i8;
    type Acc = i32;

    /// Starts a fresh per-vector accumulator for this distance kernel.
    ///
    /// See [`DistanceKernelBackend::init`] for the shared contract.
    #[inline(always)]
    fn init() -> Self::Acc {
        0
    }

    /// Accumulates one kernel chunk without reducing the partial distance.
    ///
    /// # Safety
    ///
    /// The caller must uphold [`DistanceKernelBackend::step`]'s pointer, shape and lifetime contract.
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const i8, query_chunk: *const i8) {
        for i in 0..32 {
            *acc = acc.wrapping_add((*base_chunk.add(i) as i32) * (*query_chunk.add(i) as i32));
        }
    }

    /// Converts the completed accumulator to a smaller-is-closer distance.
    ///
    /// See [`DistanceKernelBackend::reduce`] for the shared contract.
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        -(acc as f32)
    }

    /// Combines independent chunk accumulators before the final reduction.
    ///
    /// See [`DistanceKernelBackend::merge`] for the shared contract.
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into = into.wrapping_add(src);
    }
}

impl DistanceKernelBackend for IpI16Distance {
    const MAX_COORDINATES: usize = (i64::MAX as u64 / (32768_u64 * 32768)) as usize;
    type Storage = i16;
    type Acc = i64;

    /// Starts a fresh per-vector accumulator for this distance kernel.
    ///
    /// See [`DistanceKernelBackend::init`] for the shared contract.
    #[inline(always)]
    fn init() -> Self::Acc {
        0
    }

    /// Accumulates one kernel chunk without reducing the partial distance.
    ///
    /// # Safety
    ///
    /// The caller must uphold [`DistanceKernelBackend::step`]'s pointer, shape and lifetime contract.
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const i16, query_chunk: *const i16) {
        for i in 0..16 {
            *acc = acc.wrapping_add((*base_chunk.add(i) as i64) * (*query_chunk.add(i) as i64));
        }
    }

    /// Converts the completed accumulator to a smaller-is-closer distance.
    ///
    /// See [`DistanceKernelBackend::reduce`] for the shared contract.
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        -(acc as f32)
    }

    /// Combines independent chunk accumulators before the final reduction.
    ///
    /// See [`DistanceKernelBackend::merge`] for the shared contract.
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into = into.wrapping_add(src);
    }
}

impl DistanceKernelBackend for IpF32Distance {
    type Storage = f32;
    type Acc = f32;

    /// Starts a fresh per-vector accumulator for this distance kernel.
    ///
    /// See [`DistanceKernelBackend::init`] for the shared contract.
    #[inline(always)]
    fn init() -> Self::Acc {
        0.0
    }

    /// Accumulates one kernel chunk without reducing the partial distance.
    ///
    /// # Safety
    ///
    /// The caller must uphold [`DistanceKernelBackend::step`]'s pointer, shape and lifetime contract.
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const f32, query_chunk: *const f32) {
        for i in 0..8 {
            *acc += *base_chunk.add(i) * *query_chunk.add(i);
        }
    }

    /// Converts the completed accumulator to a smaller-is-closer distance.
    ///
    /// See [`DistanceKernelBackend::reduce`] for the shared contract.
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        -acc
    }

    /// Combines independent chunk accumulators before the final reduction.
    ///
    /// See [`DistanceKernelBackend::merge`] for the shared contract.
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into += src;
    }
}

impl DistanceKernelBackend for L2F32Distance {
    type Storage = f32;
    type Acc = f32;

    /// Starts a fresh per-vector accumulator for this distance kernel.
    ///
    /// See [`DistanceKernelBackend::init`] for the shared contract.
    #[inline(always)]
    fn init() -> Self::Acc {
        0.0
    }

    /// Accumulates one kernel chunk without reducing the partial distance.
    ///
    /// # Safety
    ///
    /// The caller must uphold [`DistanceKernelBackend::step`]'s pointer, shape and lifetime contract.
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const f32, query_chunk: *const f32) {
        for i in 0..8 {
            let d = *base_chunk.add(i) - *query_chunk.add(i);
            *acc += d * d;
        }
    }

    /// Converts the completed accumulator to a smaller-is-closer distance.
    ///
    /// See [`DistanceKernelBackend::reduce`] for the shared contract.
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        acc
    }

    /// Combines independent chunk accumulators before the final reduction.
    ///
    /// See [`DistanceKernelBackend::merge`] for the shared contract.
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into += src;
    }
}

// Unsigned byte products widen before accumulation. No signed-byte reinterpretation
// or saturating pairwise multiply; high-dimensional rows retain their full sum.
/// Private scalar implementation of the unsigned-byte inner-product stream kernel.
///
/// Use the public distance facade so CPU capability checks precede execution.
pub struct IpU8Distance;
impl DistanceKernelBackend for IpU8Distance {
    type Storage = u8;
    type Acc = u64;
    const CHUNK_BYTES: usize = 32;
    const MAX_COORDINATES: usize = if usize::BITS < 64 {
        usize::MAX
    } else {
        (u64::MAX / 65025) as usize
    };

    /// Starts a fresh per-vector accumulator for this distance kernel.
    ///
    /// See [`DistanceKernelBackend::init`] for the shared contract.
    #[inline(always)]
    fn init() -> Self::Acc {
        0
    }

    /// Accumulates one kernel chunk without reducing the partial distance.
    ///
    /// # Safety
    ///
    /// The caller must uphold [`DistanceKernelBackend::step`]'s pointer, shape and lifetime contract.
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, a: *const u8, b: *const u8) {
        for i in 0..32 {
            *acc += u64::from(*a.add(i)) * u64::from(*b.add(i));
        }
    }

    /// Converts the completed accumulator to a smaller-is-closer distance.
    ///
    /// See [`DistanceKernelBackend::reduce`] for the shared contract.
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        -(acc as f32)
    }

    /// Combines independent chunk accumulators before the final reduction.
    ///
    /// See [`DistanceKernelBackend::merge`] for the shared contract.
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into += src;
    }
}
