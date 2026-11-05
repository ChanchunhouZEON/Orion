/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! CPU selection lives outside the streaming loop. DistanceIsa kernels stay private:
//! their safe trait methods may contain instructions requiring the checked ISA.
//! Never expose these types as user-selectable DistanceKernelBackend implementations.

use crate::{DistanceKernelBackend, DistanceStream};
use std::sync::OnceLock;
#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512;
pub(crate) mod compute;
#[cfg(target_arch = "aarch64")]
pub(crate) mod dotprod;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(target_arch = "x86_64")]
pub(crate) mod popcnt;
pub(crate) mod scalar;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DistanceIsa {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "aarch64")]
    Dotprod,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx512Popcnt,
}

impl DistanceIsa {
    fn supported(self) -> bool {
        match self {
            Self::Scalar => true,
            #[cfg(target_arch = "aarch64")]
            Self::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(target_arch = "aarch64")]
            Self::Dotprod => {
                std::arch::is_aarch64_feature_detected!("neon")
                    && std::arch::is_aarch64_feature_detected!("dotprod")
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => {
                std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => {
                Self::Avx2.supported()
                    && std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
                    && std::is_x86_feature_detected!("avx512dq")
                    && std::is_x86_feature_detected!("avx512vl")
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx512Popcnt => {
                Self::Avx512.supported() && std::is_x86_feature_detected!("avx512vpopcntdq")
            }
        }
    }
}

#[inline]
pub(crate) fn detected_backend() -> DistanceIsa {
    static BACKEND: OnceLock<DistanceIsa> = OnceLock::new();
    *BACKEND.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        for backend in [
            DistanceIsa::Avx512Popcnt,
            DistanceIsa::Avx512,
            DistanceIsa::Avx2,
        ] {
            if backend.supported() {
                return backend;
            }
        }
        #[cfg(target_arch = "aarch64")]
        for backend in [DistanceIsa::Dotprod, DistanceIsa::Neon] {
            if backend.supported() {
                return backend;
            }
        }
        DistanceIsa::Scalar
    })
}

// Keep fallback loop bodies out of the small selection wrapper. Otherwise
// their size can make LLVM outline the dispatcher as an extra call layer.
#[inline(never)]
fn enter_scalar<K: DistanceKernelBackend, S: FnMut(u32, f32)>(
    stream: DistanceStream<'_, K>,
    sink: S,
) {
    stream.run_static(sink);
}
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn enter_neon<K: DistanceKernelBackend, S: FnMut(u32, f32)>(
    stream: DistanceStream<'_, K>,
    sink: S,
) {
    stream.run_static(sink);
}

// These are the ISA boundaries. run_static, the compute body and the sink are
// inlined together here, so no ISA checks or indirect calls occur per chunk.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn enter_avx2<K: DistanceKernelBackend, S: FnMut(u32, f32)>(
    stream: DistanceStream<'_, K>,
    sink: S,
) {
    stream.run_static(sink);
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512dq,avx512vl")]
unsafe fn enter_avx512<K: DistanceKernelBackend, S: FnMut(u32, f32)>(
    stream: DistanceStream<'_, K>,
    sink: S,
) {
    stream.run_static(sink);
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512dq,avx512vl,avx512vpopcntdq")]
unsafe fn enter_popcnt<K: DistanceKernelBackend, S: FnMut(u32, f32)>(
    stream: DistanceStream<'_, K>,
    sink: S,
) {
    stream.run_static(sink);
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn enter_dotprod<K: DistanceKernelBackend, S: FnMut(u32, f32)>(
    stream: DistanceStream<'_, K>,
    sink: S,
) {
    stream.run_static(sink);
}

macro_rules! dispatch {
    ($run:ident, $metric:ident, $storage:ty) => {
        /// Caller must have checked the selected ISA. No row layout is changed.
        #[inline(always)]
        pub(crate) unsafe fn $run<
            K: DistanceKernelBackend<Storage = $storage>,
            S: FnMut(u32, f32),
        >(
            stream: DistanceStream<'_, K>,
            sink: S,
            backend: DistanceIsa,
        ) {
            // A 64-byte backend cannot read a 32-byte padded tail. Select the
            // narrower kernel for the entire stream; never pad/repack in search.
            #[cfg(target_arch = "x86_64")]
            let backend = if matches!(backend, DistanceIsa::Avx512 | DistanceIsa::Avx512Popcnt)
                && !stream.compute_bytes().is_multiple_of(64)
            {
                DistanceIsa::Avx2
            } else {
                backend
            };
            #[cfg(target_arch = "aarch64")]
            let backend = if !stream
                .compute_bytes()
                .is_multiple_of(neon::$metric::CHUNK_BYTES)
            {
                DistanceIsa::Scalar
            } else {
                backend
            };
            // SAFETY: CPU support is established by the caller; all public row
            // spans are multiples of 32, and the 64-byte case was checked above.
            unsafe {
                match backend {
                    DistanceIsa::Scalar => {
                        enter_scalar(stream.with_kernel::<scalar::$metric>(), sink)
                    }
                    #[cfg(target_arch = "aarch64")]
                    DistanceIsa::Neon => enter_neon(stream.with_kernel::<neon::$metric>(), sink),
                    #[cfg(target_arch = "aarch64")]
                    DistanceIsa::Dotprod => {
                        enter_dotprod(stream.with_kernel::<dotprod::$metric>(), sink)
                    }
                    #[cfg(target_arch = "x86_64")]
                    DistanceIsa::Avx2 => enter_avx2(stream.with_kernel::<avx2::$metric>(), sink),
                    #[cfg(target_arch = "x86_64")]
                    DistanceIsa::Avx512 => {
                        enter_avx512(stream.with_kernel::<avx512::$metric>(), sink)
                    }
                    #[cfg(target_arch = "x86_64")]
                    DistanceIsa::Avx512Popcnt => {
                        enter_popcnt(stream.with_kernel::<popcnt::$metric>(), sink)
                    }
                }
            }
        }
    };
}
dispatch!(run_l2u8distance, L2U8Distance, u8);
dispatch!(run_l2u16distance, L2U16Distance, u16);
dispatch!(run_jlhammingdistance, JLHammingDistance, u8);
dispatch!(run_ipi8distance, IpI8Distance, i8);
dispatch!(run_ipi16distance, IpI16Distance, i16);
dispatch!(run_ipf32distance, IpF32Distance, f32);
dispatch!(run_l2f32distance, L2F32Distance, f32);

#[cfg(test)]
mod tests;
