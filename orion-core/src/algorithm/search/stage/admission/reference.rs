/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::{AdmissionSession, AdmissionStage, Neighbor};
use orion_ann_diskann::model::PaddedDataset;

/// Reference f32 admission used by matched-state experiments.
#[derive(Debug)]
pub struct ExactAdmission<'a>(pub &'a PaddedDataset<f32>);
struct ExactSession<'a> {
    data: &'a PaddedDataset<f32>,
    query: Vec<f32>,
}
// SAFETY: the borrowed padded dataset supplies checked rows of the declared shape.
unsafe impl AdmissionStage for ExactAdmission<'_> {
    fn shape(&self) -> Option<(usize, usize)> {
        Some((self.0.layout().logical_dim(), self.0.len()))
    }
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
    fn entry_distance(&self, id: u32) -> f32 {
        orion_ann_vector::distance_l2_slice_f32(&self.query, self.data.row(id as usize).unwrap())
    }
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
