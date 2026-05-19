/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Fixed random orthogonal rotation for ADSampling.
//!
//! Rotation is the pre-condition for the ADSampling scaled-partial test: after
//! applying an orthogonal `R` to all vectors, the squared-difference
//! contribution per dimension is approximately i.i.d. regardless of the
//! dataset's original axis correlations. L2 distances are preserved exactly
//! (`‖R(a-b)‖² = ‖a-b‖²`), so graph construction on rotated vectors yields the
//! same graph — only the distance-compute kernel changes at search time.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

/// Random orthogonal `N × N` rotation matrix. Built once per index, shared by
/// all queries. Held in a `Box` so a `Rotator<960>` (≈3.6 MB for GIST) does
/// not blow the stack.
pub struct Rotator<const N: usize> {
    /// Row-major N×N matrix. `apply(v)[i] = Σⱼ matrix[i*N + j] * v[j]`.
    matrix: Box<[f32]>,
}

impl<const N: usize> Rotator<N> {
    /// Build a random orthogonal matrix via Gram-Schmidt on a Gaussian seed.
    ///
    /// `seed` fixes the rotation so the same index + query pair reproduces
    /// byte-identical results across runs.
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        // Start from an N×N standard-Gaussian matrix, then orthogonalize rows.
        let mut m = vec![0.0f32; N * N];
        for slot in m.iter_mut() {
            // Box-Muller for a standard-normal sample.
            let u1: f32 = rng.random_range(f32::EPSILON..1.0f32);
            let u2: f32 = rng.random_range(0.0f32..1.0f32);
            let r: f32 = (-2.0f32 * u1.ln()).sqrt();
            let theta: f32 = 2.0f32 * std::f32::consts::PI * u2;
            *slot = r * theta.cos();
        }

        // Modified Gram-Schmidt across rows. O(N³) but runs once at build.
        for i in 0..N {
            // Orthogonalize row i against rows 0..i.
            for j in 0..i {
                let mut dot = 0.0f32;
                for k in 0..N {
                    dot += m[i * N + k] * m[j * N + k];
                }
                for k in 0..N {
                    m[i * N + k] -= dot * m[j * N + k];
                }
            }
            // Normalize row i.
            let mut norm_sq = 0.0f32;
            for k in 0..N {
                norm_sq += m[i * N + k] * m[i * N + k];
            }
            let inv = 1.0 / norm_sq.sqrt();
            for k in 0..N {
                m[i * N + k] *= inv;
            }
        }

        Self {
            matrix: m.into_boxed_slice(),
        }
    }

    /// Apply rotation: `out = R × v`. Not rayon-parallelized — callers should
    /// invoke this in a hot query loop; for one-shot dataset rotation use
    /// [`apply_batch_inplace`].
    #[inline]
    pub fn apply(&self, v: &[f32; N]) -> [f32; N] {
        let mut out = [0.0f32; N];
        for i in 0..N {
            let row = &self.matrix[i * N..(i + 1) * N];
            let mut acc = 0.0f32;
            for j in 0..N {
                acc += row[j] * v[j];
            }
            out[i] = acc;
        }
        out
    }

    /// Rotate every `N`-length chunk of a packed row-major dataset in place.
    /// Panics if `data.len()` is not a multiple of `N`.
    pub fn apply_batch_inplace(&self, data: &mut [f32]) {
        assert!(data.len() % N == 0, "data.len() must be multiple of N");
        data.par_chunks_mut(N).for_each(|chunk| {
            let mut v = [0.0f32; N];
            v.copy_from_slice(chunk);
            let rotated = self.apply(&v);
            chunk.copy_from_slice(&rotated);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_preserves_l2_distance() {
        let r = Rotator::<16>::new(42);
        let a = [
            1.0, -2.0, 3.0, 0.5, 4.0, -1.5, 2.0, 0.0, 0.7, -0.3, 1.1, 2.2, -0.9, 3.3, -2.1, 0.4,
        ];
        let b = [
            -1.0, 1.0, 0.5, 2.5, -3.0, 0.0, 1.0, 4.0, -0.7, 0.3, -1.1, -2.2, 0.9, -3.3, 2.1, -0.4,
        ];

        let ra = r.apply(&a);
        let rb = r.apply(&b);

        let d_before: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        let d_after: f32 = ra.iter().zip(rb.iter()).map(|(x, y)| (x - y).powi(2)).sum();

        assert!(
            (d_before - d_after).abs() < 1e-3,
            "L2 distance must be rotation-invariant: {d_before} vs {d_after}"
        );
    }

    #[test]
    fn rotation_norm_preserved() {
        let r = Rotator::<8>::new(123);
        let v = [0.3, -1.2, 0.5, 2.0, -0.7, 1.4, 0.1, -0.8];
        let rv = r.apply(&v);
        let n_before: f32 = v.iter().map(|x| x * x).sum();
        let n_after: f32 = rv.iter().map(|x| x * x).sum();
        assert!((n_before - n_after).abs() < 1e-4);
    }

    #[test]
    fn batch_matches_single_apply() {
        let r = Rotator::<8>::new(7);
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0];
        let expected_a = r.apply(&a);
        let expected_b = r.apply(&b);

        let mut packed = Vec::with_capacity(16);
        packed.extend_from_slice(&a);
        packed.extend_from_slice(&b);
        r.apply_batch_inplace(&mut packed);

        for i in 0..8 {
            assert!((packed[i] - expected_a[i]).abs() < 1e-5);
            assert!((packed[8 + i] - expected_b[i]).abs() < 1e-5);
        }
    }
}
