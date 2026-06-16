/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! JL Hadamard prefilter adapter — drives the
//! [`crate::model::dataset::jl_hadamard_dataset::JlHadamardDataset`]
//! through the same `JLHammingDistance` streaming kernel as the JL
//! Sparse prefilter (the byte layout is identical), but with denser
//! per-bit information from the `HDHDHD` encoder.

use super::{PrefilterSession, PrefilterStage};
use crate::model::dataset::jl_hadamard_dataset::JlHadamardDataset;

/// Wraps a [`JlHadamardDataset`] for use as a [`super::PrefilterStage`].
pub struct JlHadamardPrefilter<'a, const N: usize> {
    pub dataset: &'a JlHadamardDataset<N, 1024>,
}

impl<'a, const N: usize> JlHadamardPrefilter<'a, N> {
    #[inline]
    pub fn new(dataset: &'a JlHadamardDataset<N, 1024>) -> Self {
        Self { dataset }
    }
}

impl<'a, const N: usize> PrefilterStage<N> for JlHadamardPrefilter<'a, N> {
    #[inline]
    fn open<'b>(&'b self, q: &[f32; N]) -> Box<dyn PrefilterSession + 'b> {
        Box::new(JlHadamardSession {
            dataset: self.dataset,
            query: self.dataset.encode_query(q),
        })
    }
}

/// Per-query JL Hadamard session.
pub struct JlHadamardSession<'a, const N: usize> {
    dataset: &'a JlHadamardDataset<N, 1024>,
    query: Box<[u8]>,
}

impl<'a, const N: usize> PrefilterSession for JlHadamardSession<'a, N> {
    /// In-place cmov-compact via `DistanceStream<JLHammingDistance, N>`.
    /// Same kernel as the JL Sparse prefilter — byte layout matches
    /// (LSB-first bit-packed sign codes).
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
        let stride_bytes = self.dataset.stride;
        let base_ptr = self.dataset.codes.as_slice().as_ptr();
        let query_ptr = self.query.as_ptr();

        let id_in_ptr = id_scratch.as_ptr();
        let id_out_ptr = id_scratch.as_mut_ptr();
        let id_in = unsafe { std::slice::from_raw_parts(id_in_ptr, n_before) };

        let mut w = 0usize;
        unsafe {
            vector::DistanceStream::<vector::JLHammingDistance, N>::new(
                base_ptr,
                stride_bytes,
                stride_bytes,
                query_ptr,
                id_in,
                lookahead_lines,
            )
            .run(|id, jl_dist| {
                *id_out_ptr.add(w) = id;
                w += (jl_dist < threshold) as usize;
            });
        }
        id_scratch.truncate(w);
    }

    /// Scalar Hamming for the threshold-mean recompute.
    #[inline]
    fn distance(&self, vid: u32) -> f32 {
        self.dataset.hamming(&self.query, vid) as f32
    }
}
