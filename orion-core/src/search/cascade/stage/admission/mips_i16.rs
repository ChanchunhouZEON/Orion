/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! i16 MIPS admission tier for high-recall MIPS workloads.

use super::{AdmissionSession, AdmissionStage};
use crate::model::{MipsI16, Neighbor, ScalarQuantizedDataset};

/// Wraps a `ScalarQuantizedDataset<MipsI16>` for use as an
/// `AdmissionStage`. Twin of [`super::mips_i8::MipsI8Admission`]
/// with i16 storage for the high-recall band where i8 quantization
/// noise caps recall.
pub struct MipsI16Admission<'a> {
    dataset: &'a ScalarQuantizedDataset<MipsI16>,
}

impl<'a> MipsI16Admission<'a> {
    #[inline]
    pub fn new(dataset: &'a ScalarQuantizedDataset<MipsI16>) -> Self {
        Self { dataset }
    }
}

// SAFETY: shape describes the borrowed base; sessions own checked query storage
// and cannot outlive that base.
unsafe impl<'a> AdmissionStage for MipsI16Admission<'a> {
    fn precision(
        &self,
    ) -> Option<crate::model::dataset::quantized_dataset::capabilities::Precision> {
        use crate::model::dataset::quantized_dataset::QuantizedKind as K;
        use crate::model::dataset::quantized_dataset::capabilities::Precision as P;
        Some(P::Quantized(K::MipsI16))
    }

    fn shape(&self) -> Option<(usize, usize)> {
        Some((self.dataset.dimension, self.dataset.n))
    }
    #[inline]
    fn open<'b>(&'b self, q: &[f32]) -> Box<dyn AdmissionSession + 'b> {
        Box::new(MipsI16Session {
            dataset: self.dataset,
            padded: self.dataset.quantize_query_padded(q),
        })
    }
}

/// Per-query MIPS i16 session.
struct MipsI16Session<'a> {
    dataset: &'a ScalarQuantizedDataset<MipsI16>,
    padded: Vec<i16>,
}

// SAFETY: streams retain input IDs, write only within the input-sized output
// region, and return the initialized compacted prefix length.
unsafe impl<'a> AdmissionSession for MipsI16Session<'a> {
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
        let stride_bytes = self.dataset.stride * std::mem::size_of::<i16>();
        let base_ptr = self.dataset.data.as_slice().as_ptr() as *const u8;
        let query_ptr = self.padded.as_ptr() as *const u8;

        let mut w = 0usize;
        unsafe {
            orion_ann_vector::DistanceStream::<orion_ann_vector::IpI16Distance>::new(
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
impl std::fmt::Debug for MipsI16Admission<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MipsI16Admission")
            .field("dimension", &self.dataset.dimension)
            .field("points", &self.dataset.n)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for MipsI16Session<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MipsI16Session")
            .field("dimension", &self.dataset.dimension)
            .field("points", &self.dataset.n)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orion_ann_diskann::model::InmemDataset;

    #[test]
    fn mips_i16_session_smoke() {
        let mut ds = InmemDataset::<f32, 16>::new(4, 1.0).unwrap();
        for (i, v) in ds.data.as_mut_slice().iter_mut().enumerate() {
            *v = ((i % 7) as f32) * 0.1 - 0.3;
        }
        let qds = ScalarQuantizedDataset::<MipsI16>::from_f32_dataset(&ds);
        let adm = MipsI16Admission::new(&qds);
        let q = [0.1f32; 16];
        let session = adm.open(&q);
        for vid in 0..4u32 {
            let d = session.entry_distance(vid);
            assert!(d.is_finite(), "vid={vid} d={d}");
        }
    }
}
