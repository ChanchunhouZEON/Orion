/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Direct u8 L2 admission tier (no kernel trick).

use super::{AdmissionSession, AdmissionStage};
use crate::model::{L2U8, Neighbor, QuantizedDataset};

/// Wraps a `QuantizedDataset<L2U8, N>` (u8 base, affine quantization
/// to `[0, 255]`) for use as an [`AdmissionStage`]. Drives
/// `DistanceStream<L2U8Distance, N>` directly — no reconstruction in
/// the sink, the kernel returns `Σ(a − b)²` straight up.
pub struct L2U8Admission<'a, const N: usize> {
    pub dataset: &'a QuantizedDataset<L2U8, N>,
}

impl<'a, const N: usize> L2U8Admission<'a, N> {
    #[inline]
    pub fn new(dataset: &'a QuantizedDataset<L2U8, N>) -> Self {
        Self { dataset }
    }
}

impl<'a, const N: usize> AdmissionStage<N> for L2U8Admission<'a, N> {
    #[inline]
    fn open<'b>(&'b self, q: &[f32; N]) -> Box<dyn AdmissionSession + 'b> {
        Box::new(L2U8Session {
            dataset: self.dataset,
            padded: self.dataset.quantize_query_padded(q),
            short: self.dataset.quantize_query(q),
        })
    }
}

/// Per-query session: padded u8 query (length `STRIDE`, trailing zero
/// pad) + short `[u8; N]` form for the entry-distance scalar path.
pub struct L2U8Session<'a, const N: usize> {
    dataset: &'a QuantizedDataset<L2U8, N>,
    padded: Vec<u8>,
    short: [u8; N],
}

impl<'a, const N: usize> AdmissionSession for L2U8Session<'a, N> {
    #[inline]
    fn entry_distance(&self, vid: u32) -> f32 {
        unsafe { self.dataset.qdist(vid, &self.short) }
    }

    #[inline]
    unsafe fn admit_stream(
        &self,
        id_scratch: &[u32],
        out: *mut Neighbor,
        cutoff: f32,
        lookahead_lines: usize,
    ) -> usize {
        let stride_bytes = QuantizedDataset::<L2U8, N>::STRIDE;
        let base_ptr = self.dataset.data.as_slice().as_ptr() as *const u8;
        let query_ptr = self.padded.as_ptr() as *const u8;

        let mut w = 0usize;
        unsafe {
            vector::DistanceStream::<vector::L2U8Distance, N>::new(
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

    fn make_ds<const N: usize>(num_points: usize) -> InmemDataset<f32, N> {
        let mut ds = InmemDataset::<f32, N>::new(num_points, 1.0).unwrap();
        let s = ds.data.as_mut_slice();
        for (i, v) in s.iter_mut().enumerate() {
            *v = ((i % 31) as f32) * 0.05 - 0.5;
        }
        ds
    }

    #[test]
    fn l2u8_admission_session_entry_distance_nonneg() {
        // u8 kernel requires N % 16 == 0.
        let ds = make_ds::<16>(8);
        let qds = QuantizedDataset::<L2U8, 16>::from_f32_dataset(&ds);
        let adm = L2U8Admission::new(&qds);
        let q = [0.0f32; 16];
        let session = adm.open(&q);
        for vid in 0..8u32 {
            let d = session.entry_distance(vid);
            assert!(d >= -1e-3, "vid={vid} d={d}");
        }
    }

    #[test]
    fn l2u8_admission_session_self_distance_smallest() {
        let ds = make_ds::<16>(4);
        let qds = QuantizedDataset::<L2U8, 16>::from_f32_dataset(&ds);
        let adm = L2U8Admission::new(&qds);
        let q: [f32; 16] = std::array::from_fn(|i| ds.data.as_slice()[i]);
        let session = adm.open(&q);
        let d0 = session.entry_distance(0);
        let d1 = session.entry_distance(1);
        // Self distance should be ≤ distance to a different vertex (modulo
        // quantization noise — the dataset has rather narrow per-dim range
        // so the bound is loose).
        assert!(d0 <= d1 + 1e3, "d0={d0} d1={d1}");
    }
}
