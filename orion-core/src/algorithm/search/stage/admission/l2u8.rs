/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Direct u8 L2 admission tier (no kernel trick).

use super::{AdmissionSession, AdmissionStage};
use crate::model::{L2U8, Neighbor, ScalarQuantizedDataset};

/// Wraps a `ScalarQuantizedDataset<L2U8>` (u8 base, affine quantization
/// to `[0, 255]`) for use as an `AdmissionStage`. Drives
/// `DistanceStream<L2U8Distance>` directly — no reconstruction in
/// the sink, the kernel returns `Σ(a − b)²` straight up.
pub struct L2U8Admission<'a> {
    dataset: &'a ScalarQuantizedDataset<L2U8>,
}

impl<'a> L2U8Admission<'a> {
    #[inline]
    pub fn new(dataset: &'a ScalarQuantizedDataset<L2U8>) -> Self {
        Self { dataset }
    }
}

// SAFETY: shape describes the borrowed base; sessions own checked query storage
// and cannot outlive that base.
unsafe impl<'a> AdmissionStage for L2U8Admission<'a> {
    fn precision(
        &self,
    ) -> Option<crate::model::dataset::quantized_dataset::capabilities::Precision> {
        use crate::model::dataset::quantized_dataset::QuantizedKind as K;
        use crate::model::dataset::quantized_dataset::capabilities::Precision as P;
        Some(P::Quantized(K::L2U8))
    }

    fn shape(&self) -> Option<(usize, usize)> {
        Some((self.dataset.dimension, self.dataset.n))
    }
    #[inline]
    fn open<'b>(&'b self, q: &[f32]) -> Box<dyn AdmissionSession + 'b> {
        Box::new(L2U8Session {
            dataset: self.dataset,
            padded: self.dataset.quantize_query_padded(q),
        })
    }
}

/// Per-query session: padded u8 query (length `STRIDE`, trailing zero
/// pad) + short `[u8; N]` form for the entry-distance scalar path.
struct L2U8Session<'a> {
    dataset: &'a ScalarQuantizedDataset<L2U8>,
    padded: Vec<u8>,
}

// SAFETY: streams retain input IDs, write only within the input-sized output
// region, and return the initialized compacted prefix length.
unsafe impl<'a> AdmissionSession for L2U8Session<'a> {
    #[inline]
    fn entry_distance(&self, vid: u32) -> f32 {
        assert!(
            (vid as usize) < self.dataset.n,
            "sidecar row ID out of bounds"
        );
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
        let stride_bytes = self.dataset.stride;
        let base_ptr = self.dataset.data.as_slice().as_ptr() as *const u8;
        let query_ptr = self.padded.as_ptr() as *const u8;

        let mut w = 0usize;
        unsafe {
            orion_ann_vector::DistanceStream::<orion_ann_vector::L2U8Distance>::new(
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

// Keep diagnostics bounded: do not format vector payloads or working buffers.
impl std::fmt::Debug for L2U8Admission<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("L2U8Admission")
            .field("dimension", &self.dataset.dimension)
            .field("points", &self.dataset.n)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for L2U8Session<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("L2U8Session")
            .field("dimension", &self.dataset.dimension)
            .field("points", &self.dataset.n)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orion_ann_diskann::model::InmemDataset;

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
        let qds = ScalarQuantizedDataset::<L2U8>::from_f32_dataset(&ds);
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
        let qds = ScalarQuantizedDataset::<L2U8>::from_f32_dataset(&ds);
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
