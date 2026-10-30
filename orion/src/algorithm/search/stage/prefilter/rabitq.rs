/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! RaBitQ-as-prefilter — uses RaBitQ's rotated sign-bit code as a
//! cheap Hamming-distance prefilter ahead of the L2 admission tier.
//!
//! Same kernel as [`super::jl::JlPrefilter`] (XOR + popcount via
//! `JLHammingDistance`) but the signature comes from RaBitQ's
//! orthogonal rotation rather than JL's sparse projection. Each bit
//! summarises a rotation row over **all `N` dims**, vs JL's `NZ`
//! random dims, so per-bit discrimination is meaningfully tighter at
//! the same byte width on high-D.
//!
//! Signatures retain their packed disk layout; in memory each row is padded
//! to a complete Hamming-kernel block. The existing recipe policy still skips
//! filtering when the historical code stride is not a multiple of 32 bytes
//! (for example SIFT-128). Padding must not silently enable a new filter.
//!
//! Stage-1 setup cost: one `O(N²)` rotation per query (~µs at D=960).

use super::{PrefilterSession, PrefilterStage};
use crate::model::dataset::rabitq_dataset::RabitQDataset;

/// Wraps a [`RabitQDataset`] for use as a [`super::PrefilterStage`].
///
/// Thin reference-holding factory; per-query rotation + sign-pack
/// happens inside [`RabitqSession::open`].
pub struct RabitqPrefilter<'a> {
    pub dataset: &'a RabitQDataset,
}

impl<'a> RabitqPrefilter<'a> {
    #[inline]
    pub fn new(dataset: &'a RabitQDataset) -> Self {
        Self { dataset }
    }
}

impl<'a> PrefilterStage for RabitqPrefilter<'a> {
    fn shape(&self) -> Option<(usize, usize)> {
        Some((self.dataset.dimension, self.dataset.num_vertices))
    }
    #[inline]
    fn open<'b>(&'b self, q: &[f32]) -> Box<dyn PrefilterSession + 'b> {
        // 1. Rotate the query (O(N²)).
        let mut rotated = vec![0.0f32; self.dataset.dimension];
        self.dataset.rotate_query(q, &mut rotated);

        // 2. Sign-pack into the stride-sized buffer using the same
        //    LSB-first-per-byte convention as the base codes (bit 1 =
        //    positive, bit 0 = negative).
        let stride = self.dataset.stride;
        let mut query_bits = vec![0u8; stride].into_boxed_slice();
        for d in 0..self.dataset.dimension {
            if rotated[d] > 0.0 {
                query_bits[d >> 3] |= 1u8 << (d & 7);
            }
        }

        Box::new(RabitqSession {
            dataset: self.dataset,
            query_bits,
            stride,
        })
    }
}

/// Per-query RaBitQ-prefilter session.
pub struct RabitqSession<'a> {
    dataset: &'a RabitQDataset,
    query_bits: Box<[u8]>,
    stride: usize,
}

impl<'a> PrefilterSession for RabitqSession<'a> {
    /// In-place cmov-compact via `DistanceStream<JLHammingDistance>`
    /// — same shape as the JL prefilter, since the byte layout (bit-
    /// packed sign codes, LSB-first per byte) matches exactly.
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

        // Preserve the original low-dimensional bypass policy independently
        // of architecture-specific SIMD padding.
        if crate::model::dataset::rabitq_dataset::rabitq_code_stride_disk(self.dataset.dimension)
            % 32
            != 0
        {
            return;
        }

        let base_ptr = self.dataset.codes.as_slice().as_ptr();
        let query_ptr = self.query_bits.as_ptr();
        let id_in_ptr = id_scratch.as_ptr();
        let id_out_ptr = id_scratch.as_mut_ptr();
        let id_in = unsafe { std::slice::from_raw_parts(id_in_ptr, n_before) };

        let mut w = 0usize;
        unsafe {
            vector::DistanceStream::<vector::JLHammingDistance>::new(
                base_ptr,
                self.stride,
                self.stride,
                query_ptr,
                id_in,
                lookahead_lines,
            )
            .run(|id, ham_dist| {
                *id_out_ptr.add(w) = id;
                w += (ham_dist < threshold) as usize;
            });
        }
        id_scratch.truncate(w);
    }

    /// Scalar Hamming distance for the threshold-mean recompute.
    #[inline]
    fn distance(&self, vid: u32) -> f32 {
        let off = (vid as usize) * self.stride;
        let v_sig = &self.dataset.codes.as_slice()[off..off + self.stride];
        let mut acc: u32 = 0;
        for i in 0..self.stride {
            acc += (v_sig[i] ^ self.query_bits[i]).count_ones();
        }
        acc as f32
    }
}

#[cfg(test)]
mod tests {
    use super::super::PrefilterStage;
    use super::*;
    use diskann::model::InmemDataset;

    #[test]
    fn rabitq_prefilter_session_smoke() {
        let mut ds = InmemDataset::<f32, 32>::new(4, 1.0).unwrap();
        for (i, v) in ds.data.as_mut_slice().iter_mut().enumerate() {
            *v = ((i % 13) as f32) * 0.07 - 0.4;
        }
        let rds = RabitQDataset::build_from(&ds, 0x42);
        let pf = RabitqPrefilter::new(&rds);
        let q = [0.1f32; 32];
        let session = pf.open(&q);
        // Distance to each vertex should be a finite popcount sum.
        for vid in 0..4u32 {
            let d = session.distance(vid);
            assert!(d >= 0.0 && d.is_finite());
        }
    }
}
