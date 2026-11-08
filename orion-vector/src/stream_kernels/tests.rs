/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::*;
use crate::DistanceStream;

fn backends() -> Vec<DistanceIsa> {
    let mut all = vec![DistanceIsa::Scalar];
    #[cfg(target_arch = "aarch64")]
    all.extend([DistanceIsa::Neon, DistanceIsa::Dotprod]);
    #[cfg(target_arch = "x86_64")]
    all.extend([
        DistanceIsa::Avx2,
        DistanceIsa::Avx512,
        DistanceIsa::Avx512Popcnt,
    ]);
    all.retain(|b| b.supported());
    all
}

// Unlike a canary, PROT_NONE catches reads beyond the final padded row. A
// 32-byte row immediately before this page also exercises wide-backend fallback.
#[cfg(unix)]
struct Guarded<T> {
    mapping: *mut libc::c_void,
    bytes: usize,
    start: *mut T,
}
#[cfg(unix)]
impl<T: Copy> Guarded<T> {
    fn new(values: &[T]) -> Self {
        unsafe {
            let page = libc::sysconf(libc::_SC_PAGESIZE) as usize;
            let len = std::mem::size_of_val(values);
            let usable = len.div_ceil(page).max(1) * page;
            let mapping = libc::mmap(
                std::ptr::null_mut(),
                usable + page,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            );
            assert_ne!(mapping, libc::MAP_FAILED);
            assert_eq!(
                libc::mprotect(
                    mapping.cast::<u8>().add(usable).cast(),
                    page,
                    libc::PROT_NONE
                ),
                0
            );
            let start = mapping.cast::<u8>().add(usable - len).cast::<T>();
            std::ptr::copy_nonoverlapping(values.as_ptr(), start, values.len());
            Self {
                mapping,
                bytes: usable + page,
                start,
            }
        }
    }
    fn as_ptr(&self) -> *const T {
        self.start
    }
}
#[cfg(unix)]
impl<T> Drop for Guarded<T> {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.mapping, self.bytes);
        }
    }
}
#[cfg(not(unix))]
struct Guarded<T>(Vec<T>);
#[cfg(not(unix))]
impl<T: Copy> Guarded<T> {
    fn new(values: &[T]) -> Self {
        Self(values.to_vec())
    }
    fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }
}

macro_rules! agreement {
    ($test:ident, $metric:ident, $dispatch:ident, $t:ty, $value:expr, $term:expr, $sign:expr, $tolerance:expr) => {
        #[test]
        fn $test() {
            eprintln!("{} backends: {:?}", stringify!($metric), backends());
            for dim in [
                1, 7, 15, 16, 17, 25, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 960, 1536,
            ] {
                // Scalar's 32-byte contract is also a valid input to dispatch.
                // Do not use compile-time AVX512 padding to hide short tails.
                let count =
                    (dim * std::mem::size_of::<$t>()).div_ceil(32) * 32 / std::mem::size_of::<$t>();
                let mut base = vec![0 as $t; 8 * count];
                let mut query = vec![0 as $t; count];
                for row in 0..8 {
                    for i in 0..dim {
                        base[row * count + i] = ($value)(i + row * 17);
                    }
                }
                for i in 0..dim {
                    query[i] = ($value)(i + 107);
                }
                let ids = [7, 2, 0, 4, 4, 1, 3, 6];
                let expected: Vec<_> = ids
                    .iter()
                    .map(|&id| {
                        let total: f64 = (0..dim)
                            .map(|i| ($term)(base[id as usize * count + i], query[i]))
                            .sum();
                        (id, ($sign * total) as f32)
                    })
                    .collect();
                let base = Guarded::new(&base);
                let query = Guarded::new(&query);
                let aux = [0_u64; 8];
                let row_bytes = count * std::mem::size_of::<$t>();
                if row_bytes.is_multiple_of(crate::$metric::stream_block_bytes(row_bytes)) {
                    let mut automatic = Vec::new();
                    unsafe {
                        DistanceStream::<crate::$metric>::new(
                            base.as_ptr().cast(),
                            count * std::mem::size_of::<$t>(),
                            count * std::mem::size_of::<$t>(),
                            query.as_ptr().cast(),
                            &ids,
                            32,
                        )
                        .run(|id, d| automatic.push((id, d)));
                    }
                    assert_eq!(automatic.len(), expected.len());
                    for (&(id, a), &(eid, e)) in automatic.iter().zip(&expected) {
                        assert_eq!(id, eid);
                        assert!(
                            (a - e).abs() <= $tolerance * e.abs().max(1.0),
                            "automatic dim={dim}: {a} != {e}"
                        );
                    }
                }
                for backend in backends() {
                    for auxiliary in [false, true] {
                        let mut actual = Vec::new();
                        unsafe {
                            let stream = DistanceStream::<scalar::$metric>::new(
                                base.as_ptr().cast(),
                                count * std::mem::size_of::<$t>(),
                                count * std::mem::size_of::<$t>(),
                                query.as_ptr().cast(),
                                &ids,
                                32,
                            );
                            let stream = if auxiliary {
                                stream.with_aux(aux.as_ptr().cast(), 8)
                            } else {
                                stream
                            };
                            $dispatch(stream, |id, d| actual.push((id, d)), backend);
                        }
                        assert_eq!(actual.len(), expected.len());
                        for (&(id, a), &(eid, e)) in actual.iter().zip(&expected) {
                            assert_eq!(id, eid);
                            assert!(
                                (a - e).abs() <= $tolerance * e.abs().max(1.0),
                                "{backend:?} dim={dim}: {a} != {e}"
                            );
                        }
                    }
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
#[test]
fn short_rows_use_avx2_without_changing_wide_stream_selection() {
    for wide in [DistanceIsa::Avx512, DistanceIsa::Avx512Popcnt] {
        assert_eq!(wide.for_compute_bytes(32), DistanceIsa::Avx2);
        assert_eq!(wide.for_compute_bytes(96), DistanceIsa::Avx2);
        for bytes in [64, 128, 256, 3840, 6144] {
            assert_eq!(wide.for_compute_bytes(bytes), wide);
        }
    }
    for narrow in [DistanceIsa::Scalar, DistanceIsa::Avx2] {
        for bytes in [32, 64, 96, 128] {
            assert_eq!(narrow.for_compute_bytes(bytes), narrow);
        }
    }
}
agreement!(
    u8_backends,
    L2U8Distance,
    run_l2u8distance,
    u8,
    |i: usize| [0, 255, 1, 254, 128, 7, 23][i % 7],
    |a: u8, b: u8| (a as f64 - b as f64).powi(2),
    1.0,
    0.0
);
agreement!(
    u16_backends,
    L2U16Distance,
    run_l2u16distance,
    u16,
    |i: usize| [0, 65535, 1, 65534, 32768, 7, 12345][i % 7],
    |a: u16, b: u16| (a as f64 - b as f64).powi(2),
    1.0,
    0.0
);
agreement!(
    i8_backends,
    IpI8Distance,
    run_ipi8distance,
    i8,
    |i: usize| [-128, 127, -1, 0, 1, 64, -43][i % 7],
    |a: i8, b: i8| a as f64 * b as f64,
    -1.0,
    0.0
);
agreement!(
    i16_backends,
    IpI16Distance,
    run_ipi16distance,
    i16,
    |i: usize| [-32768, 32767, -1, 0, 1, 16384, -12345][i % 7],
    |a: i16, b: i16| a as f64 * b as f64,
    -1.0,
    0.0
);
agreement!(
    hamming_backends,
    JLHammingDistance,
    run_jlhammingdistance,
    u8,
    |i: usize| [0, 255, 0x55, 0xAA, 0xFE, 0x80, 1][i % 7],
    |a: u8, b: u8| (a ^ b).count_ones() as f64,
    1.0,
    0.0
);
agreement!(
    f32_l2_backends,
    L2F32Distance,
    run_l2f32distance,
    f32,
    |i: usize| ((i * 31 % 257) as f32 - 128.0) / 17.0,
    |a: f32, b: f32| (a as f64 - b as f64).powi(2),
    1.0,
    2e-5
);
agreement!(
    f32_ip_backends,
    IpF32Distance,
    run_ipf32distance,
    f32,
    |i: usize| ((i * 31 % 257) as f32 - 128.0) / 17.0,
    |a: f32, b: f32| a as f64 * b as f64,
    -1.0,
    2e-5
);

#[test]
fn automatic_backend_is_supported_and_empty_stream_is_safe() {
    assert!(detected_backend().supported());
    // CI and hardware validation can require coverage rather than silently
    // passing when an intended ISA is unavailable. This is test-only: production
    // callers cannot force unsupported instructions through an environment flag.
    if let Ok(required) = std::env::var("ORION_TEST_REQUIRED_BACKENDS") {
        let available: Vec<_> = backends()
            .iter()
            .map(|backend| format!("{backend:?}").to_ascii_lowercase())
            .collect();
        for name in required.split(',') {
            let normalized = name.trim().replace('-', "").to_ascii_lowercase();
            assert!(
                available.contains(&normalized),
                "required backend {name:?} unavailable; executed backends: {available:?}"
            );
        }
    }
    unsafe {
        DistanceStream::<crate::L2U8Distance>::new(
            std::ptr::null(),
            0,
            0,
            std::ptr::null(),
            &[],
            32,
        )
        .run(|_, _| panic!("empty sink"));
    }
}
