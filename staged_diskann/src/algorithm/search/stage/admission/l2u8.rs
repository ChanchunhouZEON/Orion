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
