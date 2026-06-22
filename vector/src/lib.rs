/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#![cfg_attr(
    not(test),
    warn(clippy::panic, clippy::unwrap_used, clippy::expect_used)
)]

mod distance;
mod half;
mod metric;
mod utils;
mod vector_storage;

// ───────────────────────── Architecture dispatch ─────────────────────────
//
// Three platform tiers, mutually exclusive:
//   * `target_arch = "aarch64"`                             — NEON kernels
//   * `target_arch = "x86_64" + target_feature = "avx512f"` — AVX-512 kernels
//   * everything else                                       — scalar fallback
//
// The NEON file (`l2_neon_distance`, `ip_neon_distance`, etc.) carries
// both the NEON impl (gated `cfg(target_arch = "aarch64")` inside the
// file) AND the scalar fallback (gated `cfg(not(target_arch =
// "aarch64"))` inside the file). On x86_64 with AVX-512, we skip the
// NEON-file module entirely so its scalar-fallback branch doesn't
// collide with the AVX-512 file's `distance_*` exports.
//
// See `.cargo/config.toml.example` for the AVX-512 build recipe.

// Existing AVX2 L2 helper — kept for the historical x86_64 path that
// pre-dates the AVX-512 work. Not re-exported at top level; only used
// internally by `distance.rs` on x86_64 without AVX-512.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
mod l2_float_distance;

// AVX-512 path: only loaded when both arch and target_feature match.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod ip_avx512_distance;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod ip_avx512_distance_i16;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod ip_avx512_distance_i8;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod l2_avx512_distance;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod l2_avx512_distance_u8;

// NEON / scalar-fallback path: every target EXCEPT x86_64+AVX-512.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
mod ip_neon_distance;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
mod ip_neon_distance_i16;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
mod ip_neon_distance_i8;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
mod l2_neon_distance;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
mod l2_neon_distance_u8;

// Top-level re-exports route to the right module per target.
//
// `distance_l2_vector_f32` lives in:
//   * `l2_neon_distance.rs` for aarch64 (NEON impl only — no scalar
//     fallback in this file; the scalar f32 path is in
//     `l2_scalar_distance.rs`).
//   * `l2_avx512_distance.rs` for x86_64+AVX-512.
//   * `l2_float_distance.rs` for x86_64 without AVX-512 (AVX2 path).
//   * `l2_scalar_distance.rs` for everything else.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use l2_avx512_distance::distance_l2_vector_f32;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub use l2_float_distance::distance_l2_vector_f32;
#[cfg(target_arch = "aarch64")]
pub use l2_neon_distance::distance_l2_vector_f32;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub use l2_scalar_distance::distance_l2_vector_f32;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use l2_avx512_distance_u8::{distance_l2_vector_u8, distance_l2_vector_u8_batch4};
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub use l2_neon_distance_u8::{distance_l2_vector_u8, distance_l2_vector_u8_batch4};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use ip_avx512_distance::{distance_ip_vector_f32, distance_ip_vector_f32_batch4};
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub use ip_neon_distance::{distance_ip_vector_f32, distance_ip_vector_f32_batch4};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use ip_avx512_distance_i8::{distance_ip_vector_i8, distance_ip_vector_i8_batch4};
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub use ip_neon_distance_i8::{distance_ip_vector_i8, distance_ip_vector_i8_batch4};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use ip_avx512_distance_i16::{distance_ip_vector_i16, distance_ip_vector_i16_batch4};
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub use ip_neon_distance_i16::{distance_ip_vector_i16, distance_ip_vector_i16_batch4};

mod distance_fn;
pub use distance_fn::{
    DistanceFn, IpF32Distance, IpI16Distance, IpI8Distance, JLHammingDistance, L2F32Distance,
    L2U16Distance, L2U8Distance,
};

mod distance_buffer;
pub use distance_buffer::CacheLineDistanceBuffer;

mod distance_stream;
pub use distance_stream::DistanceStream;

// Scalar fallback for architectures without SIMD support
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod l2_scalar_distance;

pub use crate::half::Half;
pub use distance::FullPrecisionDistance;
pub use metric::Metric;
pub use utils::{prefetch_vector, CACHE_LINE_BYTES};
pub use vector_storage::VectorStorage;

#[cfg(test)]
mod test_util;

#[cfg(test)]
mod distance_test {
    use super::*;
    use crate::test_util::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dist_l2_f32_small() {
        // Use a dimension divisible by both 4 (NEON) and 8 (AVX2)
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let dist =
            <[f32; 8] as FullPrecisionDistance<f32, 8>>::distance_compare(&a, &b, Metric::L2);
        // Each dimension differs by 1.0, so L2 = 8 * 1.0^2 = 8.0
        assert_abs_diff_eq!(dist, 8.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dist_l2_f32_zeros() {
        let a = [0.0f32; 8];
        let b = [0.0f32; 8];
        let dist =
            <[f32; 8] as FullPrecisionDistance<f32, 8>>::distance_compare(&a, &b, Metric::L2);
        assert_abs_diff_eq!(dist, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dist_l2_f32_matches_scalar() {
        let a: [f32; 16] = [
            1.5, 2.3, -0.7, 4.1, 0.0, -3.2, 8.8, 1.0, -2.5, 6.7, 0.1, -1.1, 3.3, 9.9, -4.4, 2.2,
        ];
        let b: [f32; 16] = [
            -0.5, 1.1, 2.3, -1.0, 5.5, 0.8, -2.1, 3.7, 4.4, -0.3, 7.7, 2.2, -1.8, 0.6, 5.5, -3.3,
        ];

        let simd_dist =
            <[f32; 16] as FullPrecisionDistance<f32, 16>>::distance_compare(&a, &b, Metric::L2);
        let scalar_dist = no_vector_compare_f32(&a, &b);
        assert_abs_diff_eq!(simd_dist, scalar_dist, epsilon = 1e-4);
    }

    #[test]
    fn test_adsampling_matches_full_when_under_bound() {
        let a: [f32; 128] = std::array::from_fn(|i| (i as f32).sin());
        let b: [f32; 128] = std::array::from_fn(|i| (i as f32).cos());
        let full =
            <[f32; 128] as FullPrecisionDistance<f32, 128>>::distance_compare(&a, &b, Metric::L2);
        // Large bound — no abandonment, should return full distance.
        let ads = <[f32; 128] as FullPrecisionDistance<f32, 128>>::distance_compare_adsampling(
            &a,
            &b,
            full * 10.0,
            2.1,
        );
        assert!((ads - full).abs() < 1e-3, "ads={ads} full={full}");
    }

    #[test]
    fn test_adsampling_abandons_when_distant() {
        let a: [f32; 128] = std::array::from_fn(|i| i as f32);
        let b: [f32; 128] = std::array::from_fn(|i| -(i as f32));
        // Tiny bound — should abandon early and return -1.0.
        let ads = <[f32; 128] as FullPrecisionDistance<f32, 128>>::distance_compare_adsampling(
            &a, &b, 1.0, 2.1,
        );
        assert!(ads < 0.0, "expected abandonment, got {ads}");
    }

    #[test]
    fn test_dist_l2_f16_matches_scalar() {
        let a_f32: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b_f32: [f32; 8] = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let a: [Half; 8] = std::array::from_fn(|i| Half::from_f32(a_f32[i]));
        let b: [Half; 8] = std::array::from_fn(|i| Half::from_f32(b_f32[i]));

        let dist =
            <[Half; 8] as FullPrecisionDistance<Half, 8>>::distance_compare(&a, &b, Metric::L2);
        let scalar_dist = no_vector_compare_f16(&a, &b);
        assert_eq!(dist, scalar_dist);
    }

    // ─── Helpers ─────────────────────────────────────────────────────

    /// Deterministic xorshift64 so tests reproduce exactly.
    fn rng_next(state: &mut u64) -> u64 {
        let mut s = *state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *state = s;
        s
    }

    fn rand_f32(state: &mut u64, lo: f32, hi: f32) -> f32 {
        let u = (rng_next(state) >> 32) as u32;
        let unit = (u as f32) / (u32::MAX as f32);
        lo + (hi - lo) * unit
    }

    // ─── IP f32 ──────────────────────────────────────────────────────

    #[test]
    fn test_ip_f32_matches_scalar() {
        const N: usize = 128;
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        let a: [f32; N] = std::array::from_fn(|_| rand_f32(&mut s, -1.0, 1.0));
        let b: [f32; N] = std::array::from_fn(|_| rand_f32(&mut s, -1.0, 1.0));
        let simd = distance_ip_vector_f32::<N>(&a, &b);
        let scalar = scalar_ip_f32(&a, &b);
        assert_abs_diff_eq!(simd, scalar, epsilon = 1e-3);
    }

    #[test]
    fn test_ip_f32_batch4_matches_single() {
        const N: usize = 64;
        let mut s = 0xDEADBEEF_CAFEBABEu64;
        let a0: [f32; N] = std::array::from_fn(|_| rand_f32(&mut s, -1.0, 1.0));
        let a1: [f32; N] = std::array::from_fn(|_| rand_f32(&mut s, -1.0, 1.0));
        let a2: [f32; N] = std::array::from_fn(|_| rand_f32(&mut s, -1.0, 1.0));
        let a3: [f32; N] = std::array::from_fn(|_| rand_f32(&mut s, -1.0, 1.0));
        let q: [f32; N] = std::array::from_fn(|_| rand_f32(&mut s, -1.0, 1.0));
        let batch = distance_ip_vector_f32_batch4::<N>(&a0, &a1, &a2, &a3, &q);
        let expected = [
            distance_ip_vector_f32::<N>(&a0, &q),
            distance_ip_vector_f32::<N>(&a1, &q),
            distance_ip_vector_f32::<N>(&a2, &q),
            distance_ip_vector_f32::<N>(&a3, &q),
        ];
        for i in 0..4 {
            assert_abs_diff_eq!(batch[i], expected[i], epsilon = 1e-4);
        }
    }

    // ─── L2 u8 ───────────────────────────────────────────────────────

    #[test]
    fn test_l2_u8_matches_scalar() {
        const N: usize = 128;
        let mut s = 0xA5A5_5A5A_F00D_BEEFu64;
        let a: [u8; N] = std::array::from_fn(|_| (rng_next(&mut s) & 0xFF) as u8);
        let b: [u8; N] = std::array::from_fn(|_| (rng_next(&mut s) & 0xFF) as u8);
        let simd = distance_l2_vector_u8::<N>(&a, &b);
        let scalar = scalar_l2_u8(&a, &b);
        assert_eq!(simd as u32, scalar as u32);
    }

    #[test]
    fn test_l2_u8_batch4_matches_single() {
        const N: usize = 32;
        let mut s = 0x0123_4567_89AB_CDEFu64;
        let a0: [u8; N] = std::array::from_fn(|_| (rng_next(&mut s) & 0xFF) as u8);
        let a1: [u8; N] = std::array::from_fn(|_| (rng_next(&mut s) & 0xFF) as u8);
        let a2: [u8; N] = std::array::from_fn(|_| (rng_next(&mut s) & 0xFF) as u8);
        let a3: [u8; N] = std::array::from_fn(|_| (rng_next(&mut s) & 0xFF) as u8);
        let q: [u8; N] = std::array::from_fn(|_| (rng_next(&mut s) & 0xFF) as u8);
        let batch = distance_l2_vector_u8_batch4::<N>(&a0, &a1, &a2, &a3, &q);
        let expected = [
            distance_l2_vector_u8::<N>(&a0, &q),
            distance_l2_vector_u8::<N>(&a1, &q),
            distance_l2_vector_u8::<N>(&a2, &q),
            distance_l2_vector_u8::<N>(&a3, &q),
        ];
        for i in 0..4 {
            assert_eq!(batch[i] as u32, expected[i] as u32);
        }
    }

    // ─── IP i8 ───────────────────────────────────────────────────────

    #[test]
    fn test_ip_i8_matches_scalar() {
        const N: usize = 128;
        let mut s = 0xFEEDFACE_BAADF00Du64;
        let a: [i8; N] = std::array::from_fn(|_| (rng_next(&mut s) & 0xFF) as i8);
        let b: [i8; N] = std::array::from_fn(|_| (rng_next(&mut s) & 0xFF) as i8);
        let simd = distance_ip_vector_i8::<N>(&a, &b);
        let scalar = scalar_ip_i8(&a, &b);
        assert_eq!(simd, scalar);
    }

    #[test]
    fn test_ip_i8_batch4_matches_single() {
        const N: usize = 64;
        let mut s = 0x4242_4242_4242_4242u64;
        let a0: [i8; N] = std::array::from_fn(|_| (rng_next(&mut s) & 0xFF) as i8);
        let a1: [i8; N] = std::array::from_fn(|_| (rng_next(&mut s) & 0xFF) as i8);
        let a2: [i8; N] = std::array::from_fn(|_| (rng_next(&mut s) & 0xFF) as i8);
        let a3: [i8; N] = std::array::from_fn(|_| (rng_next(&mut s) & 0xFF) as i8);
        let q: [i8; N] = std::array::from_fn(|_| (rng_next(&mut s) & 0xFF) as i8);
        let batch = distance_ip_vector_i8_batch4::<N>(&a0, &a1, &a2, &a3, &q);
        let expected = [
            distance_ip_vector_i8::<N>(&a0, &q),
            distance_ip_vector_i8::<N>(&a1, &q),
            distance_ip_vector_i8::<N>(&a2, &q),
            distance_ip_vector_i8::<N>(&a3, &q),
        ];
        for i in 0..4 {
            assert_eq!(batch[i], expected[i]);
        }
    }

    // ─── IP i16 ──────────────────────────────────────────────────────

    #[test]
    fn test_ip_i16_matches_scalar() {
        const N: usize = 64;
        let mut s = 0xC0DE_C0DE_FEED_BEEFu64;
        // Clamp to ±10000 so the i64 accumulator stays well-bounded.
        let a: [i16; N] = std::array::from_fn(|_| {
            let v = (rng_next(&mut s) as i32) % 10_000;
            v as i16
        });
        let b: [i16; N] = std::array::from_fn(|_| {
            let v = (rng_next(&mut s) as i32) % 10_000;
            v as i16
        });
        let simd = distance_ip_vector_i16::<N>(&a, &b);
        let scalar = scalar_ip_i16(&a, &b);
        assert_eq!(simd, scalar);
    }

    #[test]
    fn test_ip_i16_batch4_matches_single() {
        const N: usize = 32;
        let mut s = 0xD0D0_BABA_F00D_F11Eu64;
        let a0: [i16; N] = std::array::from_fn(|_| (rng_next(&mut s) as i32 % 10_000) as i16);
        let a1: [i16; N] = std::array::from_fn(|_| (rng_next(&mut s) as i32 % 10_000) as i16);
        let a2: [i16; N] = std::array::from_fn(|_| (rng_next(&mut s) as i32 % 10_000) as i16);
        let a3: [i16; N] = std::array::from_fn(|_| (rng_next(&mut s) as i32 % 10_000) as i16);
        let q: [i16; N] = std::array::from_fn(|_| (rng_next(&mut s) as i32 % 10_000) as i16);
        let batch = distance_ip_vector_i16_batch4::<N>(&a0, &a1, &a2, &a3, &q);
        let expected = [
            distance_ip_vector_i16::<N>(&a0, &q),
            distance_ip_vector_i16::<N>(&a1, &q),
            distance_ip_vector_i16::<N>(&a2, &q),
            distance_ip_vector_i16::<N>(&a3, &q),
        ];
        for i in 0..4 {
            assert_eq!(batch[i], expected[i]);
        }
    }

    // ─── Identity / commutativity sanity checks ─────────────────────

    #[test]
    fn test_l2_f32_self_distance_zero() {
        let a: [f32; 64] = std::array::from_fn(|i| i as f32 * 0.5);
        let d = distance_l2_vector_f32(&a, &a);
        assert_abs_diff_eq!(d, 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_ip_f32_self_negative_norm_sq() {
        // <a,a> = ||a||² so -<a,a> = -||a||²
        let a: [f32; 64] = std::array::from_fn(|i| (i as f32 * 0.1).cos());
        let d = distance_ip_vector_f32(&a, &a);
        let norm_sq: f32 = a.iter().map(|x| x * x).sum();
        assert_abs_diff_eq!(d, -norm_sq, epsilon = 1e-3);
    }

    #[test]
    fn test_l2_u8_self_distance_zero() {
        let a: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(7));
        let d = distance_l2_vector_u8(&a, &a);
        assert_eq!(d, 0.0);
    }

    // ─── Metric enum + Half scalar cast ─────────────────────────────

    #[test]
    fn test_metric_enum_basics() {
        use std::str::FromStr;
        let m = Metric::L2;
        assert_eq!(format!("{m:?}"), "L2");
        let m2 = m;
        assert!(matches!(m2, Metric::L2));
        assert_eq!(Metric::from_str("l2").unwrap(), Metric::L2);
        assert_eq!(Metric::from_str("L2").unwrap(), Metric::L2);
        assert_eq!(Metric::from_str("cosine").unwrap(), Metric::Cosine);
        assert!(Metric::from_str("unknown").is_err());
        // PartialEq + Clone derive coverage.
        assert_eq!(Metric::Cosine, Metric::Cosine.clone());
    }

    #[test]
    fn test_half_roundtrip_within_range() {
        for v in [0.0f32, 1.0, -1.0, 100.0, -100.0, 0.5, -0.25] {
            let h = Half::from_f32(v);
            let back = h.to_f32();
            assert_abs_diff_eq!(back, v, epsilon = 0.01);
        }
    }
}
