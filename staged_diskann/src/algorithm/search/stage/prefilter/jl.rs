/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! JL Sparse 1024-bit Hamming prefilter adapters.
//!
//! Two distinct adapter types — one per metric, one struct each:
//!
//! * [`JlPrefilter`] (L2): wraps [`JLSparseDataset`]. The
//!   per-candidate score is raw Hamming distance — a noisy estimate
//!   of the angle, which on the L2 unit-sphere collapses to a noisy
//!   L2 estimate. Mirrors PA's `Euclidean_JL_Sparse_Point`.
//! * [`JlMipsPrefilter`] (MIPS): wraps [`JLSparseDatasetMips`]. The
//!   per-candidate score is `popcount × ‖v‖`, mirroring PA's
//!   `Mips_JL_Sparse_Point_Normalized::distance` (`jl_point.h:319`).
//!   The base norm `‖v‖` is read from the dataset's per-vertex
//!   `norms` sidecar. The query norm `‖q‖` is dropped from the
//!   formula (PA does the same — `jl_point.h:319` comments out the
//!   `* qr` factor) because it is a per-query constant that does not
//!   affect ranking.
//!
//! The two adapters are intentionally separate types so the L2 path
//! pays nothing for the unused norms slab and the MIPS path can keep
//! its norms in cache-aligned storage right next to the codes.

use super::{PrefilterSession, PrefilterStage};
use crate::model::{JLSparseDataset, JLSparseDatasetMips};

// ── L2 variant ────────────────────────────────────────────────────────

/// L2-mode JL prefilter wrapping a [`JLSparseDataset`] (no per-vertex
/// norms). Const-generic over `NZ` so the L2 default (`NZ=9`) can
/// stay tunable without breaking the type.
pub struct JlPrefilter<'a, const N: usize, const NZ: usize = 9> {
    pub dataset: &'a JLSparseDataset<N, 1024, NZ>,
}

impl<'a, const N: usize, const NZ: usize> JlPrefilter<'a, N, NZ> {
    #[inline]
    pub fn new(dataset: &'a JLSparseDataset<N, 1024, NZ>) -> Self {
        Self { dataset }
    }
}

impl<'a, const N: usize, const NZ: usize> PrefilterStage<N> for JlPrefilter<'a, N, NZ> {
    #[inline]
    fn open<'b>(&'b self, q: &[f32; N]) -> Box<dyn PrefilterSession + 'b> {
        Box::new(JlSession {
            dataset: self.dataset,
            query: self.dataset.encode_query(q),
        })
    }
}

pub struct JlSession<'a, const N: usize, const NZ: usize> {
    dataset: &'a JLSparseDataset<N, 1024, NZ>,
    query: Box<[u8]>,
}

impl<'a, const N: usize, const NZ: usize> PrefilterSession for JlSession<'a, N, NZ> {
    /// In-place cmov-compact `id_scratch` via
    /// `DistanceStream<JLHammingDistance, N>`. Same shape as the
    /// legacy `search_l2_u8_q` JL block — STRIDE = 128 B (BITS=1024)
    /// so each vertex is exactly one cache line, hitting hot path A
    /// in `DistanceStream::resolve`.
    #[inline]
    unsafe fn filter_compact(
        &self,
        id_scratch: &mut Vec<u32>,
        threshold: f32,
        lookahead_lines: usize,
    ) {
        let n_before = id_scratch.len();
        if n_before == 0 {
            return;
        }
        let q_jl_stride_bytes = self.dataset.stride;
        let q_jl_base_ptr = self.dataset.codes.as_slice().as_ptr();
        let q_jl_query_ptr = self.query.as_ptr();

        let id_in_ptr = id_scratch.as_ptr();
        let id_out_ptr = id_scratch.as_mut_ptr();
        let id_in = unsafe { std::slice::from_raw_parts(id_in_ptr, n_before) };

        let mut w = 0usize;
        unsafe {
            vector::DistanceStream::<vector::JLHammingDistance, N>::new(
                q_jl_base_ptr,
                q_jl_stride_bytes,
                q_jl_stride_bytes,
                q_jl_query_ptr,
                id_in,
                lookahead_lines,
            )
            .run(|id, jl_dist| {
                // `<` rather than `<=` — matches PA's `>= threshold`
                // skip rule in `beamSearch.h:171`.
                *id_out_ptr.add(w) = id;
                w += (jl_dist < threshold) as usize;
            });
        }
        id_scratch.truncate(w);
    }

    #[inline]
    fn distance(&self, vid: u32) -> f32 {
        self.dataset.hamming(&self.query, vid) as f32
    }
}

// ── MIPS variant ──────────────────────────────────────────────────────

/// MIPS-mode JL prefilter wrapping a [`JLSparseDatasetMips`]
/// (includes per-vertex `‖v‖` sidecar). Per-candidate score is
/// `popcount × ‖v‖` per PA's `Mips_JL_Sparse_Point_Normalized`.
/// Const-generic over `NZ` so the MIPS sweet-spot (currently `NZ=9`)
/// can be tuned independently of the L2 path.
pub struct JlMipsPrefilter<'a, const N: usize, const NZ: usize = 9> {
    pub dataset: &'a JLSparseDatasetMips<N, 1024, NZ>,
}

impl<'a, const N: usize, const NZ: usize> JlMipsPrefilter<'a, N, NZ> {
    #[inline]
    pub fn new(dataset: &'a JLSparseDatasetMips<N, 1024, NZ>) -> Self {
        Self { dataset }
    }
}

impl<'a, const N: usize, const NZ: usize> PrefilterStage<N> for JlMipsPrefilter<'a, N, NZ> {
    #[inline]
    fn open<'b>(&'b self, q: &[f32; N]) -> Box<dyn PrefilterSession + 'b> {
        Box::new(JlMipsSession {
            dataset: self.dataset,
            query: self.dataset.encode_query(q),
        })
    }
}

pub struct JlMipsSession<'a, const N: usize, const NZ: usize> {
    dataset: &'a JLSparseDatasetMips<N, 1024, NZ>,
    query: Box<[u8]>,
}

impl<'a, const N: usize, const NZ: usize> PrefilterSession for JlMipsSession<'a, N, NZ> {
    /// Same streaming kernel as the L2 variant, but the per-candidate
    /// score is multiplied by `‖v‖` (looked up from
    /// `dataset.norms`) before the threshold compare.
    #[inline]
    unsafe fn filter_compact(
        &self,
        id_scratch: &mut Vec<u32>,
        threshold: f32,
        lookahead_lines: usize,
    ) {
        let n_before = id_scratch.len();
        if n_before == 0 {
            return;
        }
        let q_jl_stride_bytes = self.dataset.stride;
        let q_jl_base_ptr = self.dataset.codes.as_slice().as_ptr();
        let q_jl_query_ptr = self.query.as_ptr();
        let norms_ptr = self.dataset.norms.as_slice().as_ptr();

        let id_in_ptr = id_scratch.as_ptr();
        let id_out_ptr = id_scratch.as_mut_ptr();
        let id_in = unsafe { std::slice::from_raw_parts(id_in_ptr, n_before) };

        let mut w = 0usize;
        unsafe {
            vector::DistanceStream::<vector::JLHammingDistance, N>::new(
                q_jl_base_ptr,
                q_jl_stride_bytes,
                q_jl_stride_bytes,
                q_jl_query_ptr,
                id_in,
                lookahead_lines,
            )
            .run(|id, jl_dist| {
                let score = jl_dist * *norms_ptr.add(id as usize);
                *id_out_ptr.add(w) = id;
                w += (score < threshold) as usize;
            });
        }
        id_scratch.truncate(w);
    }

    #[inline]
    fn distance(&self, vid: u32) -> f32 {
        self.dataset.mips_distance(&self.query, vid)
    }
}
