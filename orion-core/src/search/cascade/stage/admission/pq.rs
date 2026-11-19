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
    /// Binds the dataset to this stage; query encoding is deferred until the stage is opened.
    pub fn new(dataset: &'a PqDataset, metric: PqMetric) -> Self {
        Self { dataset, metric }
    }
}
// SAFETY: shape reflects the immutable codes; open validates all query inputs.
unsafe impl AdmissionStage for PqAdmission<'_> {
    /// Identifies the precision used to validate cascade ordering.
    ///
    /// See [`AdmissionStage::precision`] for the shared contract.
    fn precision(&self) -> Option<Precision> {
        Some(Precision::ProductQuantized {
            inner_product: self.metric == PqMetric::InnerProduct,
        })
    }

    /// Describes the base shape required by this admission representation.
    ///
    /// See [`AdmissionStage::shape`] for the shared contract.
    fn shape(&self) -> Option<(usize, usize)> {
        Some((self.dataset.dimension(), self.dataset.len()))
    }

    /// Prepares the query state consumed by this admission stage.
    ///
    /// See [`AdmissionStage::open`] for the shared contract.
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
    /// Scores the entry node in the admission distance domain.
    ///
    /// See [`AdmissionSession::entry_distance`] for the shared contract.
    fn entry_distance(&self, id: u32) -> f32 {
        self.distance(id).expect("validated graph ID")
    }

    /// Scores candidates and compacts qualifying neighbors into caller-owned output.
    ///
    /// # Safety
    ///
    /// The caller must uphold [`AdmissionSession::admit_stream`]'s pointer, shape and lifetime contract.
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
