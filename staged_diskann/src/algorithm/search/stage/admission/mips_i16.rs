/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! i16 MIPS admission tier for high-recall MIPS workloads.

use super::{AdmissionSession, AdmissionStage};
use crate::model::{MipsI16, Neighbor, QuantizedDataset};

/// Wraps a `QuantizedDataset<MipsI16, N>` for use as an
/// [`AdmissionStage`]. Twin of [`super::mips_i8::MipsI8Admission`]
/// with i16 storage for the high-recall band where i8 quantization
/// noise caps recall.
pub struct MipsI16Admission<'a, const N: usize> {
    pub dataset: &'a QuantizedDataset<MipsI16, N>,
}

impl<'a, const N: usize> MipsI16Admission<'a, N> {
    #[inline]
    pub fn new(dataset: &'a QuantizedDataset<MipsI16, N>) -> Self {
        Self { dataset }
    }
}

impl<'a, const N: usize> AdmissionStage<N> for MipsI16Admission<'a, N> {
    #[inline]
    fn open<'b>(&'b self, q: &[f32; N]) -> Box<dyn AdmissionSession + 'b> {
        Box::new(MipsI16Session {
            dataset: self.dataset,
            padded: self.dataset.quantize_query_padded(q),
            short: self.dataset.quantize_query(q),
        })
    }
}

/// Per-query MIPS i16 session.
pub struct MipsI16Session<'a, const N: usize> {
    dataset: &'a QuantizedDataset<MipsI16, N>,
    padded: Vec<i16>,
    short: [i16; N],
}

impl<'a, const N: usize> AdmissionSession for MipsI16Session<'a, N> {
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
        let stride_bytes = QuantizedDataset::<MipsI16, N>::STRIDE * std::mem::size_of::<i16>();
        let base_ptr = self.dataset.data.as_slice().as_ptr() as *const u8;
        let query_ptr = self.padded.as_ptr() as *const u8;

        let mut w = 0usize;
        unsafe {
            vector::DistanceStream::<vector::IpI16Distance, N>::new(
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
