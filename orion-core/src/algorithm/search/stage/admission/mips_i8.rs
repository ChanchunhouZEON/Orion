/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! i8 MIPS admission tier (`sdot` for unit-normalised data).

use super::{AdmissionSession, AdmissionStage};
use crate::model::{MipsI8, Neighbor, ScalarQuantizedDataset};

/// Wraps a `ScalarQuantizedDataset<MipsI8>` for use as an
/// `AdmissionStage`. Drives `DistanceStream<IpI8Distance>` —
/// the `sdot` kernel. Distance is returned as `-IP` so the
/// "smaller == closer" convention works directly in the PQ.
pub struct MipsI8Admission<'a> {
    dataset: &'a ScalarQuantizedDataset<MipsI8>,
}

impl<'a> MipsI8Admission<'a> {
    #[inline]
    pub fn new(dataset: &'a ScalarQuantizedDataset<MipsI8>) -> Self {
        Self { dataset }
    }
}

// SAFETY: shape describes the borrowed base; sessions own checked query storage
// and cannot outlive that base.
unsafe impl<'a> AdmissionStage for MipsI8Admission<'a> {
    fn shape(&self) -> Option<(usize, usize)> {
        Some((self.dataset.dimension, self.dataset.n))
    }
    #[inline]
    fn open<'b>(&'b self, q: &[f32]) -> Box<dyn AdmissionSession + 'b> {
        Box::new(MipsI8Session {
            dataset: self.dataset,
            padded: self.dataset.quantize_query_padded(q),
        })
    }
}

/// Per-query MIPS i8 session: padded i8 query + short `[i8; N]` form.
struct MipsI8Session<'a> {
    dataset: &'a ScalarQuantizedDataset<MipsI8>,
    padded: Vec<i8>,
}

// SAFETY: streams retain input IDs, write only within the input-sized output
// region, and return the initialized compacted prefix length.
unsafe impl<'a> AdmissionSession for MipsI8Session<'a> {
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
            orion_ann_vector::DistanceStream::<orion_ann_vector::IpI8Distance>::new(
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
impl std::fmt::Debug for MipsI8Admission<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MipsI8Admission")
            .field("dimension", &self.dataset.dimension)
            .field("points", &self.dataset.n)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for MipsI8Session<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MipsI8Session")
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
    fn mips_i8_session_smoke() {
        let mut ds = InmemDataset::<f32, 16>::new(4, 1.0).unwrap();
        for (i, v) in ds.data.as_mut_slice().iter_mut().enumerate() {
            *v = ((i % 7) as f32) * 0.1 - 0.3;
        }
        let qds = ScalarQuantizedDataset::<MipsI8>::from_f32_dataset(&ds);
        let adm = MipsI8Admission::new(&qds);
        let q = [0.1f32; 16];
        let session = adm.open(&q);
        // entry_distance must return a finite value for every vertex.
        for vid in 0..4u32 {
            let d = session.entry_distance(vid);
            assert!(d.is_finite(), "vid={vid} d={d}");
        }
    }
}
