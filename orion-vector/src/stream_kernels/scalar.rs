/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Private scalar kernels. Called only through a capability-checked stream entry.

use crate::DistanceKernelBackend;
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
    type Acc = u32;
    #[inline(always)]
    fn init() -> Self::Acc {
        0
    }
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
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        acc as f32
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into = into.wrapping_add(src);
    }
}

impl DistanceKernelBackend for L2U16Distance {
    const MAX_COORDINATES: usize = (u64::MAX / (65535_u64 * 65535)) as usize;
    type Storage = u16;
    type Acc = u64;
    #[inline(always)]
    fn init() -> Self::Acc {
        0
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u16, query_chunk: *const u16) {
        for i in 0..16 {
            let a = *base_chunk.add(i) as i32;
            let b = *query_chunk.add(i) as i32;
            let d = (a - b).unsigned_abs() as u64;
            *acc = acc.wrapping_add(d * d);
        }
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        acc as f32
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into = into.wrapping_add(src);
    }
}

impl DistanceKernelBackend for JLHammingDistance {
    type Storage = u8;
    type Acc = u32;
    #[inline(always)]
    fn init() -> Self::Acc {
        0
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u8, query_chunk: *const u8) {
        for i in 0..32 {
            *acc = acc.wrapping_add(((*base_chunk.add(i)) ^ (*query_chunk.add(i))).count_ones());
        }
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        acc as f32
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into = into.wrapping_add(src);
    }
}

impl DistanceKernelBackend for IpI8Distance {
    const MAX_COORDINATES: usize = (i32::MAX as u64 / (128 * 128)) as usize;
    type Storage = i8;
    type Acc = i32;
    #[inline(always)]
    fn init() -> Self::Acc {
        0
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const i8, query_chunk: *const i8) {
        for i in 0..32 {
            *acc = acc.wrapping_add((*base_chunk.add(i) as i32) * (*query_chunk.add(i) as i32));
        }
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        -(acc as f32)
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into = into.wrapping_add(src);
    }
}

impl DistanceKernelBackend for IpI16Distance {
    const MAX_COORDINATES: usize = (i64::MAX as u64 / (32768_u64 * 32768)) as usize;
    type Storage = i16;
    type Acc = i64;
    #[inline(always)]
    fn init() -> Self::Acc {
        0
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const i16, query_chunk: *const i16) {
        for i in 0..16 {
            *acc = acc.wrapping_add((*base_chunk.add(i) as i64) * (*query_chunk.add(i) as i64));
        }
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        -(acc as f32)
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into = into.wrapping_add(src);
    }
}

impl DistanceKernelBackend for IpF32Distance {
    type Storage = f32;
    type Acc = f32;
    #[inline(always)]
    fn init() -> Self::Acc {
        0.0
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const f32, query_chunk: *const f32) {
        for i in 0..8 {
            *acc += *base_chunk.add(i) * *query_chunk.add(i);
        }
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        -acc
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into += src;
    }
}

impl DistanceKernelBackend for L2F32Distance {
    type Storage = f32;
    type Acc = f32;
    #[inline(always)]
    fn init() -> Self::Acc {
        0.0
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const f32, query_chunk: *const f32) {
        for i in 0..8 {
            let d = *base_chunk.add(i) - *query_chunk.add(i);
            *acc += d * d;
        }
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        acc
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into += src;
    }
}

// Unsigned byte products widen before accumulation. No signed-byte reinterpretation
// or saturating pairwise multiply; high-dimensional rows retain their full sum.
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
    #[inline(always)]
    fn init() -> Self::Acc {
        0
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, a: *const u8, b: *const u8) {
        for i in 0..32 {
            *acc += u64::from(*a.add(i)) * u64::from(*b.add(i));
        }
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        -(acc as f32)
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into += src;
    }
}
