/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! i8 MIPS admission tier (`sdot` for unit-normalised data).

use super::{AdmissionSession, AdmissionStage};
use crate::model::{MipsI8, Neighbor, QuantizedDataset};

/// Wraps a `QuantizedDataset<MipsI8, N>` for use as an
/// [`AdmissionStage`]. Drives `DistanceStream<IpI8Distance, N>` —
/// the `sdot` kernel. Distance is returned as `-IP` so the
/// "smaller == closer" convention works directly in the PQ.
pub struct MipsI8Admission<'a, const N: usize> {
    pub dataset: &'a QuantizedDataset<MipsI8, N>,
}

impl<'a, const N: usize> MipsI8Admission<'a, N> {
    #[inline]
    pub fn new(dataset: &'a QuantizedDataset<MipsI8, N>) -> Self {
        Self { dataset }
    }
}

impl<'a, const N: usize> AdmissionStage<N> for MipsI8Admission<'a, N> {
    #[inline]
    fn open<'b>(&'b self, q: &[f32; N]) -> Box<dyn AdmissionSession + 'b> {
        Box::new(MipsI8Session {
            dataset: self.dataset,
            padded: self.dataset.quantize_query_padded(q),
            short: self.dataset.quantize_query(q),
        })
    }
}

/// Per-query MIPS i8 session: padded i8 query + short `[i8; N]` form.
pub struct MipsI8Session<'a, const N: usize> {
    dataset: &'a QuantizedDataset<MipsI8, N>,
    padded: Vec<i8>,
    short: [i8; N],
}

impl<'a, const N: usize> AdmissionSession for MipsI8Session<'a, N> {
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
        let stride_bytes = QuantizedDataset::<MipsI8, N>::STRIDE;
        let base_ptr = self.dataset.data.as_slice().as_ptr() as *const u8;
        let query_ptr = self.padded.as_ptr() as *const u8;

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
            .run(|id, qd| {
                out.add(w).write(Neighbor::new(id, qd));
                w += (qd < cutoff) as usize;
            });
        }
        w
    }
}
