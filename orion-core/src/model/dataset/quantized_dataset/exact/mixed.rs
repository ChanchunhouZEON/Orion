/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Layout at a glance
//! Base rows retain their original padded f32/u8 storage. One logical f64 query
//! preserves signed, fractional and wide-u32 coordinates without narrowing.
//! This correctness fallback does not change existing homogeneous SIMD streams.
use crate::{ScoreMetric, model::Neighbor, search::cascade::stage::AdmissionSession};
use bytemuck::Pod;
use orion_ann_diskann::model::PaddedDataset;

pub(crate) struct MixedSession<'a, T: Pod> {
    base: &'a PaddedDataset<T>,
    query: &'a [f64],
    metric: ScoreMetric,
}
impl<'a, T: Pod + Into<f32>> MixedSession<'a, T> {
    pub(crate) fn new(base: &'a PaddedDataset<T>, query: &'a [f64], metric: ScoreMetric) -> Self {
        Self {
            base,
            query,
            metric,
        }
    }
    fn distance(&self, id: u32) -> f32 {
        let score = crate::search::query::numeric::score(
            self.query,
            self.base.row(id as usize).expect("validated mixed ID"),
            self.metric,
        );
        match self.metric {
            ScoreMetric::SquaredL2 => score,
            ScoreMetric::InnerProduct => -score,
            ScoreMetric::CosineSimilarity => 1.0 - score,
        }
    }
}
// SAFETY: only input IDs are emitted, with at most one initialized output per
// ID. Base/query borrows remain valid for the complete session lifetime.
unsafe impl<T: Pod + Into<f32> + Send + Sync> AdmissionSession for MixedSession<'_, T> {
    fn entry_distance(&self, id: u32) -> f32 {
        self.distance(id)
    }
    unsafe fn admit_stream(
        &self,
        ids: &[u32],
        out: *mut Neighbor,
        cutoff: f32,
        _near: usize,
    ) -> usize {
        let mut count = 0;
        for &id in ids {
            let distance = self.distance(id);
            unsafe {
                out.add(count).write(Neighbor::new(id, distance));
            }
            count += (distance < cutoff) as usize;
        }
        count
    }
}
