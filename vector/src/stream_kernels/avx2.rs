//! 32-byte AVX2 kernels. Integer products widen before accumulation; no
//! saturating byte dot-product instruction is used (signed extrema are valid).
use crate::DistanceKernelBackend;
use std::arch::x86_64::*;

pub struct L2U8Distance;
pub struct L2U16Distance;
pub struct JLHammingDistance;
pub struct IpI8Distance;
pub struct IpI16Distance;
pub struct IpF32Distance;
pub struct L2F32Distance;

#[inline(always)]
unsafe fn sum_i32(x: __m256i) -> i32 {
    let x = _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
    let x = _mm_add_epi32(x, _mm_srli_si128(x, 8));
    _mm_cvtsi128_si32(_mm_add_epi32(x, _mm_srli_si128(x, 4)))
}
#[inline(always)]
unsafe fn sum_i64(x: __m256i) -> i64 {
    let x = _mm_add_epi64(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
    _mm_cvtsi128_si64(_mm_add_epi64(x, _mm_srli_si128(x, 8)))
}
#[inline(always)]
unsafe fn sum_f32(x: __m256) -> f32 {
    let x = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    let x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    _mm_cvtss_f32(_mm_add_ss(x, _mm_shuffle_ps(x, x, 0x55)))
}

macro_rules! byte_kernel {
    ($name:ident, $storage:ty, $limit:expr, $widen:ident, $l2:expr) => {
        impl DistanceKernelBackend for $name {
            type Storage = $storage;
            type Acc = __m256i;
            const MAX_COORDINATES: usize = $limit;
            #[inline(always)]
            fn init() -> Self::Acc {
                unsafe { _mm256_setzero_si256() }
            }
            #[inline(always)]
            unsafe fn step(acc: &mut Self::Acc, a: *const $storage, b: *const $storage) {
                // Two independent 16-byte loads cover exactly one 32-byte block.
                for offset in [0, 16] {
                    let x = $widen(_mm_loadu_si128(a.add(offset).cast()));
                    let y = $widen(_mm_loadu_si128(b.add(offset).cast()));
                    let product = if $l2 {
                        let d = _mm256_sub_epi16(x, y);
                        _mm256_madd_epi16(d, d)
                    } else {
                        _mm256_madd_epi16(x, y)
                    };
                    *acc = _mm256_add_epi32(*acc, product);
                }
            }
            #[inline(always)]
            fn reduce(acc: Self::Acc) -> f32 {
                unsafe {
                    if $l2 {
                        sum_i32(acc) as u32 as f32
                    } else {
                        -(sum_i32(acc) as f32)
                    }
                }
            }
            #[inline(always)]
            fn merge(into: &mut Self::Acc, src: Self::Acc) {
                unsafe {
                    *into = _mm256_add_epi32(*into, src);
                }
            }
        }
    };
}
byte_kernel!(
    L2U8Distance,
    u8,
    (u32::MAX as u64 / (255 * 255)) as usize,
    _mm256_cvtepu8_epi16,
    true
);
byte_kernel!(
    IpI8Distance,
    i8,
    (i32::MAX as u64 / (128 * 128)) as usize,
    _mm256_cvtepi8_epi16,
    false
);

macro_rules! word_kernel {
    ($name:ident, $storage:ty, $limit:expr, $widen:ident, $extend:ident, $l2:expr) => {
        impl DistanceKernelBackend for $name {
            type Storage = $storage;
            type Acc = [__m256i; 2];
            const MAX_COORDINATES: usize = $limit;
            #[inline(always)]
            fn init() -> Self::Acc {
                unsafe { [_mm256_setzero_si256(); 2] }
            }
            #[inline(always)]
            unsafe fn step(acc: &mut Self::Acc, a: *const $storage, b: *const $storage) {
                for half in 0..2 {
                    let x = $widen(_mm_loadu_si128(a.add(half * 8).cast()));
                    let y = $widen(_mm_loadu_si128(b.add(half * 8).cast()));
                    let p = if $l2 {
                        let d = _mm256_sub_epi32(x, y);
                        _mm256_mullo_epi32(d, d)
                    } else {
                        _mm256_mullo_epi32(x, y)
                    };
                    // Unsigned squares may exceed i32::MAX; signed IP must
                    // sign-extend. Both widen before any potentially overflowing sum.
                    acc[0] = _mm256_add_epi64(acc[0], $extend(_mm256_castsi256_si128(p)));
                    acc[1] = _mm256_add_epi64(acc[1], $extend(_mm256_extracti128_si256(p, 1)));
                }
            }
            #[inline(always)]
            fn reduce(acc: Self::Acc) -> f32 {
                unsafe {
                    let total = sum_i64(_mm256_add_epi64(acc[0], acc[1]));
                    if $l2 {
                        total as u64 as f32
                    } else {
                        -(total as f32)
                    }
                }
            }
            #[inline(always)]
            fn merge(into: &mut Self::Acc, src: Self::Acc) {
                unsafe {
                    for i in 0..2 {
                        into[i] = _mm256_add_epi64(into[i], src[i]);
                    }
                }
            }
        }
    };
}
word_kernel!(
    L2U16Distance,
    u16,
    (u64::MAX / (65535_u64 * 65535)) as usize,
    _mm256_cvtepu16_epi32,
    _mm256_cvtepu32_epi64,
    true
);
word_kernel!(
    IpI16Distance,
    i16,
    (i64::MAX as u64 / (32768_u64 * 32768)) as usize,
    _mm256_cvtepi16_epi32,
    _mm256_cvtepi32_epi64,
    false
);

impl DistanceKernelBackend for JLHammingDistance {
    type Storage = u8;
    type Acc = __m256i;
    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { _mm256_setzero_si256() }
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, a: *const u8, b: *const u8) {
        let x = _mm256_xor_si256(_mm256_loadu_si256(a.cast()), _mm256_loadu_si256(b.cast()));
        let lut = _mm256_broadcastsi128_si256(_mm_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        ));
        let mask = _mm256_set1_epi8(15);
        let low = _mm256_shuffle_epi8(lut, _mm256_and_si256(x, mask));
        let high = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(x, 4), mask));
        *acc = _mm256_add_epi64(
            *acc,
            _mm256_sad_epu8(_mm256_add_epi8(low, high), _mm256_setzero_si256()),
        );
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { sum_i64(acc) as f32 }
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe {
            *into = _mm256_add_epi64(*into, src);
        }
    }
}

macro_rules! float_kernel {
    ($name:ident, $l2:expr) => {
        impl DistanceKernelBackend for $name {
            type Storage = f32;
            type Acc = __m256;
            #[inline(always)]
            fn init() -> Self::Acc {
                unsafe { _mm256_setzero_ps() }
            }
            #[inline(always)]
            unsafe fn step(acc: &mut Self::Acc, a: *const f32, b: *const f32) {
                let x = _mm256_loadu_ps(a);
                let y = _mm256_loadu_ps(b);
                *acc = if $l2 {
                    let d = _mm256_sub_ps(x, y);
                    _mm256_fmadd_ps(d, d, *acc)
                } else {
                    _mm256_fmadd_ps(x, y, *acc)
                };
            }
            #[inline(always)]
            fn reduce(acc: Self::Acc) -> f32 {
                unsafe {
                    if $l2 {
                        sum_f32(acc)
                    } else {
                        -sum_f32(acc)
                    }
                }
            }
            #[inline(always)]
            fn merge(into: &mut Self::Acc, src: Self::Acc) {
                unsafe {
                    *into = _mm256_add_ps(*into, src);
                }
            }
        }
    };
}
float_kernel!(L2F32Distance, true);
float_kernel!(IpF32Distance, false);
