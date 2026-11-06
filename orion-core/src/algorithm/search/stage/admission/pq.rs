/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use super::{AdmissionSession, AdmissionStage};
use crate::model::{
    Neighbor,
    dataset::quantized_dataset::{
        PreparedQuery,
        capabilities::Precision,
        pq::{PqDataset, PqMetric, PqQuery},
    },
};

/// ADC admission with full-f32 query lookup tables and packed u8 base codes.
#[derive(Debug)]
pub struct PqAdmission<'a> {
    dataset: &'a PqDataset,
    metric: PqMetric,
}
impl<'a> PqAdmission<'a> {
    pub fn new(dataset: &'a PqDataset, metric: PqMetric) -> Self {
        Self { dataset, metric }
    }
}
// SAFETY: shape reflects the immutable codes; open validates all query inputs.
unsafe impl AdmissionStage for PqAdmission<'_> {
    fn precision(&self) -> Option<Precision> {
        Some(Precision::ProductQuantized {
            inner_product: self.metric == PqMetric::InnerProduct,
        })
    }
    fn shape(&self) -> Option<(usize, usize)> {
        Some((self.dataset.dimension(), self.dataset.len()))
    }
    fn open<'a>(&'a self, q: &[f32]) -> Box<dyn AdmissionSession + 'a> {
        Box::new(
            self.dataset
                .query(q, self.metric)
                .expect("validated finite PQ query"),
        )
    }
}
// SAFETY: the stream emits only its input IDs and initializes the compacted prefix.
unsafe impl AdmissionSession for PqQuery<'_> {
    fn entry_distance(&self, id: u32) -> f32 {
        self.distance(id).expect("validated graph ID")
    }
    unsafe fn admit_stream(
        &self,
        ids: &[u32],
        out: *mut Neighbor,
        cutoff: f32,
        lookahead: usize,
    ) -> usize {
        let mut count = 0;
        unsafe {
            self.stream_unchecked(ids, lookahead, |id, d| {
                out.add(count).write(Neighbor::new(id, d));
                count += (d < cutoff) as usize;
            });
        }
        count
    }
}
