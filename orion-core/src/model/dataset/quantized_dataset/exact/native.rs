/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Byte inner-product and cosine identity sessions. These only adapt scores;
//! traversal, query scheduling and SIMD prefetch remain shared.
use crate::model::{Neighbor, dataset::native_u8::NativeU8Norms};
use crate::search::cascade::stage::AdmissionSession;
use orion_ann_diskann::model::PaddedDataset;
use orion_ann_vector::IpU8Distance;
pub(crate) struct NativeSession<'a, const COSINE: bool> {
    pub(crate) base: &'a PaddedDataset<u8>,
    pub(crate) query: &'a PaddedDataset<u8>,
    pub(crate) norms: Option<&'a NativeU8Norms>,
    pub(crate) query_norm: f64,
}
impl<const COSINE: bool> NativeSession<'_, COSINE> {
    #[inline(always)]
    fn distance(&self, id: u32, negative_dot: f32) -> f32 {
        if COSINE {
            let n = self.norms.expect("cosine norms prepared").get(id as usize) as f64;
            // Cast before multiplication: two valid u64 norms need not have a
            // product representable as an integer. Keep cancellation in f64.
            (1.0 + negative_dot as f64 / (n.sqrt() * self.query_norm)).clamp(0.0, 1.0) as f32
        } else {
            negative_dot
        }
    }
}
// SAFETY: the stream preserves IDs and emits at most one initialized neighbor
// per supplied ID. The caller guarantees all IDs belong to the resident base.
unsafe impl<const COSINE: bool> AdmissionSession for NativeSession<'_, COSINE> {
    fn entry_distance(&self, id: u32) -> f32 {
        self.distance(
            id,
            self.base
                .distance::<IpU8Distance>(id, self.query, 0)
                .expect("validated byte layout"),
        )
    }
    unsafe fn admit_stream(
        &self,
        ids: &[u32],
        out: *mut Neighbor,
        cutoff: f32,
        near: usize,
    ) -> usize {
        let mut count = 0;
        let stream = unsafe {
            self.base
                .stream_unchecked::<IpU8Distance>(self.query, 0, ids, near)
        };
        let stream = if COSINE {
            let norms = self.norms.expect("cosine norms prepared");
            // SAFETY: one norm per valid base ID; session borrows both allocations.
            unsafe { stream.with_aux(norms.bytes().as_ptr(), norms.element_bytes()) }
        } else {
            stream
        };
        stream.run(|id, dot| {
            let distance = self.distance(id, dot);
            unsafe {
                out.add(count).write(Neighbor::new(id, distance));
            }
            count += (distance < cutoff) as usize;
        });
        count
    }
}
