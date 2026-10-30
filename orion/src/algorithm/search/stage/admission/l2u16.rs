/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! u16 L2 admission tier — PA's `-quantize_bits 16` recipe.

use super::{AdmissionSession, AdmissionStage};
use crate::model::{L2U16, Neighbor, QuantizedDataset};

/// Wraps a `QuantizedDataset<L2U16>` (u16 base, 256× finer per-dim
/// resolution than u8) for use as an [`AdmissionStage`]. Drives
/// `DistanceStream<L2U16Distance>`.
///
/// This is **PA's production admission tier** at `-quantize_bits 16
/// -quantize_mode 3`. 2× the DRAM of u8 per vertex (15 cache lines on
/// GIST D=960 vs 8 for u8), but the richer per-cmp signal lets PA
/// match recall at fewer hops + finer cutoff resolution.
pub struct L2U16Admission<'a> {
    pub dataset: &'a QuantizedDataset<L2U16>,
}

impl<'a> L2U16Admission<'a> {
    #[inline]
    pub fn new(dataset: &'a QuantizedDataset<L2U16>) -> Self {
        Self { dataset }
    }
}

impl<'a> AdmissionStage for L2U16Admission<'a> {
    fn shape(&self) -> Option<(usize, usize)> {
        Some((self.dataset.dimension, self.dataset.n))
    }
    #[inline]
    fn open<'b>(&'b self, q: &[f32]) -> Box<dyn AdmissionSession + 'b> {
        Box::new(L2U16Session {
            dataset: self.dataset,
            padded: self.dataset.quantize_query_padded(q),
        })
    }
}

/// Per-query session: padded u16 query + short `[u16; N]` form for
/// entry distance.
pub struct L2U16Session<'a> {
    dataset: &'a QuantizedDataset<L2U16>,
    padded: Vec<u16>,
}

impl<'a> AdmissionSession for L2U16Session<'a> {
    #[inline]
    fn entry_distance(&self, vid: u32) -> f32 {
        unsafe { self.dataset.qdist(vid, &self.padded) }
    }

    #[inline]
    unsafe fn admit_stream(
        &self,
        id_scratch: &[u32],
        out: *mut Neighbor,
        cutoff: f32,
        lookahead_lines: usize,
    ) -> usize {
        // STRIDE counts u16 elements; convert to bytes for the
        // DistanceStream's pointer arithmetic.
        let stride_bytes = self.dataset.stride * std::mem::size_of::<u16>();
        let base_ptr = self.dataset.data.as_slice().as_ptr() as *const u8;
        let query_ptr = self.padded.as_ptr() as *const u8;

        let mut w = 0usize;
        unsafe {
            vector::DistanceStream::<vector::L2U16Distance>::new(
                base_ptr,
                stride_bytes,
                stride_bytes,
                query_ptr,
                id_scratch,
                lookahead_lines,
            )
            .run(|id, qd| {
                out.add(w).write(Neighbor::new(id, qd));
                w += (qd < cutoff) as usize;
            });
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann::model::InmemDataset;

    #[test]
    fn l2u16_session_smoke() {
        let mut ds = InmemDataset::<f32, 16>::new(4, 1.0).unwrap();
        for (i, v) in ds.data.as_mut_slice().iter_mut().enumerate() {
            *v = ((i % 5) as f32) * 0.2 - 0.4;
        }
        let qds = QuantizedDataset::<L2U16>::from_f32_dataset(&ds);
        let adm = L2U16Admission::new(&qds);
        let q = [0.0f32; 16];
        let session = adm.open(&q);
        for vid in 0..4u32 {
            let d = session.entry_distance(vid);
            assert!(d >= -1e-3 && d.is_finite(), "vid={vid} d={d}");
        }
    }
}
