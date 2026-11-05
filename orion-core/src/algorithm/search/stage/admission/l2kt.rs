/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! L2 kernel-trick admission tier — `IpI8Distance` (sdot) + per-vert
//! `‖x_i8‖²` reconstruction. Current `search_l2_u8_q` default.

use super::{AdmissionSession, AdmissionStage};
use crate::model::{L2KTDataset, Neighbor};

/// Wraps an [`L2KTDataset`] for use as an `AdmissionStage`.
///
/// The kernel-trick identity rebuilds `‖q − x‖²` from
/// `IpI8Distance::reduce`'s `-IP`:
///
/// ```text
/// ‖q − x‖² = ‖q‖² + ‖x‖² − 2·IP
///          = q_norm_sq + x_sq + 2·neg_ip
/// ```
///
/// The sink performs the reconstruction inline; the norms slab is
/// prefetched alongside the i8 base via
/// [`orion_ann_vector::DistanceStream::with_aux`].
pub struct L2KTAdmission<'a> {
    dataset: &'a L2KTDataset,
}

impl<'a> L2KTAdmission<'a> {
    #[inline]
    pub fn new(dataset: &'a L2KTDataset) -> Self {
        Self { dataset }
    }
}

// SAFETY: shape describes the borrowed base; sessions own checked query storage
// and cannot outlive that base.
unsafe impl<'a> AdmissionStage for L2KTAdmission<'a> {
    fn shape(&self) -> Option<(usize, usize)> {
        Some((self.dataset.dimension, self.dataset.num_vertices))
    }
    #[inline]
    fn open<'b>(&'b self, q: &[f32]) -> Box<dyn AdmissionSession + 'b> {
        let padded = self.dataset.quantize_query_padded(q);
        let q_norm_sq = self.dataset.query_norm_sq(&padded) as f32;
        Box::new(L2KTSession {
            dataset: self.dataset,
            padded,
            q_norm_sq,
        })
    }
}

/// Per-query L2-KT session: padded i8 query + scalar `q_norm_sq` for
/// the sink-side reconstruction.
struct L2KTSession<'a> {
    dataset: &'a L2KTDataset,
    padded: Vec<i8>,
    q_norm_sq: f32,
}

// SAFETY: streams retain input IDs, write only within the input-sized output
// region, and return the initialized compacted prefix length.
unsafe impl<'a> AdmissionSession for L2KTSession<'a> {
    #[inline]
    fn entry_distance(&self, vid: u32) -> f32 {
        assert!(
            (vid as usize) < self.dataset.num_vertices,
            "sidecar row ID out of bounds"
        );
        use orion_ann_vector::DistanceKernelBackend;
        let stride_elems = self.dataset.stride;
        unsafe {
            let v_ptr = self
                .dataset
                .data
                .as_slice()
                .as_ptr()
                .add(vid as usize * stride_elems);
            let block = orion_ann_vector::IpI8Distance::CHUNK_BYTES;
            let chunks = stride_elems / block;
            let mut acc = orion_ann_vector::IpI8Distance::init();
            for c in 0..chunks {
                let off = c * block;
                orion_ann_vector::IpI8Distance::step(
                    &mut acc,
                    v_ptr.add(off),
                    self.padded.as_ptr().add(off),
                );
            }
            let neg_ip = orion_ann_vector::IpI8Distance::reduce(acc);
            let x_sq = *self.dataset.norms_sq.as_slice().as_ptr().add(vid as usize) as f32;
            self.q_norm_sq + x_sq + 2.0 * neg_ip
        }
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
        let x_norms_ptr = self.dataset.norms_sq.as_slice().as_ptr();
        let q_norm_sq = self.q_norm_sq;

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
            .with_aux(x_norms_ptr as *const u8, std::mem::size_of::<i32>())
            .run(|id, neg_ip| {
                let x_sq = *x_norms_ptr.add(id as usize) as f32;
                let qd = q_norm_sq + x_sq + 2.0 * neg_ip;
                out.add(w).write(Neighbor::new(id, qd));
                w += (qd < cutoff) as usize;
            });
        }
        w
    }
}

// Keep diagnostics bounded: do not format vector payloads or working buffers.
impl std::fmt::Debug for L2KTAdmission<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("L2KTAdmission")
            .field("dimension", &self.dataset.dimension)
            .field("points", &self.dataset.num_vertices)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for L2KTSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("L2KTSession")
            .field("dimension", &self.dataset.dimension)
            .field("points", &self.dataset.num_vertices)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orion_ann_diskann::model::InmemDataset;

    #[test]
    fn l2kt_session_smoke() {
        let mut ds = InmemDataset::<f32, 16>::new(4, 1.0).unwrap();
        for (i, v) in ds.data.as_mut_slice().iter_mut().enumerate() {
            *v = ((i % 11) as f32) * 0.08 - 0.4;
        }
        let kt = L2KTDataset::build_from(&ds);
        let adm = L2KTAdmission::new(&kt);
        let q = [0.0f32; 16];
        let session = adm.open(&q);
        for vid in 0..4u32 {
            let d = session.entry_distance(vid);
            assert!(d.is_finite(), "vid={vid} d={d}");
        }
    }
}
