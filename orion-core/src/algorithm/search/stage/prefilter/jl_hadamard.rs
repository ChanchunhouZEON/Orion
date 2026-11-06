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

/// Wraps a [`JlHadamardDataset`] for use as a `super::PrefilterStage`.
pub struct JlHadamardPrefilter<'a> {
    dataset: &'a JlHadamardDataset,
}

impl<'a> JlHadamardPrefilter<'a> {
    #[inline]
    pub fn new(dataset: &'a JlHadamardDataset) -> Self {
        Self { dataset }
    }
}

// SAFETY: shape describes the borrowed base; sessions own checked query storage
// and cannot outlive that base.
unsafe impl<'a> PrefilterStage for JlHadamardPrefilter<'a> {
    fn precision(
        &self,
    ) -> Option<crate::model::dataset::quantized_dataset::capabilities::Precision> {
        use crate::model::dataset::quantized_dataset::QuantizedKind as K;
        use crate::model::dataset::quantized_dataset::capabilities::Precision as P;
        Some(P::Quantized(K::JlHadamard))
    }

    fn shape(&self) -> Option<(usize, usize)> {
        Some((self.dataset.dimension, self.dataset.num_vertices))
    }
    #[inline]
    fn open<'b>(&'b self, q: &[f32]) -> Box<dyn PrefilterSession + 'b> {
        Box::new(JlHadamardSession {
            dataset: self.dataset,
            query: self.dataset.encode_query(q),
        })
    }
}

/// Per-query JL Hadamard session.
struct JlHadamardSession<'a> {
    dataset: &'a JlHadamardDataset,
    query: Box<[u8]>,
}

// SAFETY: streams retain input IDs, write only within the input-sized output
// region, and return the initialized compacted prefix length.
unsafe impl<'a> PrefilterSession for JlHadamardSession<'a> {
    /// In-place cmov-compact via `DistanceStream<JLHammingDistance>`.
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
            orion_ann_vector::DistanceStream::<orion_ann_vector::JLHammingDistance>::new(
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

// Keep diagnostics bounded: do not format vector payloads or working buffers.
impl std::fmt::Debug for JlHadamardPrefilter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JlHadamardPrefilter")
            .field("dimension", &self.dataset.dimension)
            .field("points", &self.dataset.num_vertices)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for JlHadamardSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JlHadamardSession")
            .field("dimension", &self.dataset.dimension)
            .field("points", &self.dataset.num_vertices)
            .finish_non_exhaustive()
    }
}
