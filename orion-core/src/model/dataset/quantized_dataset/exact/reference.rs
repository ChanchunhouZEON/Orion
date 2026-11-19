/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::model::Neighbor;
use crate::search::cascade::stage::{AdmissionSession, AdmissionStage};
use orion_ann_diskann::model::PaddedDataset;

/// Reference f32 admission used by matched-state experiments.
#[derive(Debug)]
pub struct ExactAdmission<'a>(
    /// Borrowed original coordinates used by the reference admission path.
    pub &'a PaddedDataset<f32>,
);

/// Borrowed full-precision query/base session for exact candidate scoring.
struct ExactSession<'a> {
    data: &'a PaddedDataset<f32>,
    query: Vec<f32>,
}
// SAFETY: the borrowed padded dataset supplies checked rows of the declared shape.
unsafe impl AdmissionStage for ExactAdmission<'_> {
    /// Identifies the precision used to validate cascade ordering.
    ///
    /// See [`AdmissionStage::precision`] for the shared contract.
    fn precision(
        &self,
    ) -> Option<crate::model::dataset::quantized_dataset::capabilities::Precision> {
        use crate::model::dataset::quantized_dataset::capabilities::Precision as P;
        Some(P::Exact)
    }

    /// Describes the base shape required by this admission representation.
    ///
    /// See [`AdmissionStage::shape`] for the shared contract.
    fn shape(&self) -> Option<(usize, usize)> {
        Some((self.0.layout().logical_dim(), self.0.len()))
    }

    /// Prepares the query state consumed by this admission stage.
    ///
    /// See [`AdmissionStage::open`] for the shared contract.
    fn open<'a>(&'a self, q: &[f32]) -> Box<dyn AdmissionSession + 'a> {
        assert_eq!(q.len(), self.0.layout().logical_dim());
        Box::new(ExactSession {
            data: self.0,
            query: q.to_vec(),
        })
    }
}
// SAFETY: each emitted slot is initialized with an ID from the input.
unsafe impl AdmissionSession for ExactSession<'_> {
    /// Scores the entry node in the admission distance domain.
    ///
    /// See [`AdmissionSession::entry_distance`] for the shared contract.
    fn entry_distance(&self, id: u32) -> f32 {
        orion_ann_vector::distance_l2_slice_f32(&self.query, self.data.row(id as usize).unwrap())
    }

    /// Scores candidates and compacts qualifying neighbors into caller-owned output.
    ///
    /// # Safety
    ///
    /// The caller must uphold [`AdmissionSession::admit_stream`]'s pointer, shape and lifetime contract.
    unsafe fn admit_stream(&self, ids: &[u32], out: *mut Neighbor, cutoff: f32, _: usize) -> usize {
        let mut n = 0;
        for &id in ids {
            let d = self.entry_distance(id);
            if d < cutoff {
                unsafe {
                    out.add(n).write(Neighbor::new(id, d));
                }
                n += 1;
            }
        }
        n
    }
}
