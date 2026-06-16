/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! L2 kernel-trick admission tier — `IpI8Distance` (sdot) + per-vert
//! `‖x_i8‖²` reconstruction. Current `search_l2_u8_q` default.

use super::{AdmissionSession, AdmissionStage};
use crate::model::{L2KTDataset, Neighbor};

/// Wraps an [`L2KTDataset`] for use as an [`AdmissionStage`].
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
/// [`vector::DistanceStream::with_aux`].
pub struct L2KTAdmission<'a, const N: usize> {
    pub dataset: &'a L2KTDataset<N>,
}

impl<'a, const N: usize> L2KTAdmission<'a, N> {
    #[inline]
    pub fn new(dataset: &'a L2KTDataset<N>) -> Self {
        Self { dataset }
    }
}

impl<'a, const N: usize> AdmissionStage<N> for L2KTAdmission<'a, N> {
    #[inline]
    fn open<'b>(&'b self, q: &[f32; N]) -> Box<dyn AdmissionSession + 'b> {
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
pub struct L2KTSession<'a, const N: usize> {
    dataset: &'a L2KTDataset<N>,
    padded: Vec<i8>,
    q_norm_sq: f32,
}

impl<'a, const N: usize> AdmissionSession for L2KTSession<'a, N> {
    #[inline]
    fn entry_distance(&self, vid: u32) -> f32 {
        use vector::DistanceFn;
        let stride_elems = L2KTDataset::<N>::STRIDE;
        unsafe {
            let v_ptr = self
                .dataset
                .data
                .as_slice()
                .as_ptr()
                .add(vid as usize * stride_elems);
            let chunks = stride_elems / 32;
            let mut acc = vector::IpI8Distance::init();
            for c in 0..chunks {
                let off = c * 32;
                vector::IpI8Distance::step(&mut acc, v_ptr.add(off), self.padded.as_ptr().add(off));
            }
            let neg_ip = vector::IpI8Distance::reduce(acc);
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
        let stride_bytes = L2KTDataset::<N>::STRIDE;
        let base_ptr = self.dataset.data.as_slice().as_ptr() as *const u8;
        let query_ptr = self.padded.as_ptr() as *const u8;
        let x_norms_ptr = self.dataset.norms_sq.as_slice().as_ptr();
        let q_norm_sq = self.q_norm_sq;

        let mut w = 0usize;
        unsafe {
            vector::DistanceStream::<vector::IpI8Distance, N>::new(
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
