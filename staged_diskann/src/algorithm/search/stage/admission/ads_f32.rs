/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! ADSampling f32 admission tier.
//!
//! Uses the scaled-partial-sum L2 kernel from [`vector`]
//! (`distance_compare_adsampling`) — chunked early-abort against the
//! per-hop PQ cutoff. The kernel returns `-1.0` when the partial sum
//! crosses the confidence band before completion, and the L2² sum
//! otherwise. cmov-compact admits on `dist ≥ 0 && dist < cutoff`.
//!
//! Unlike the i8 / i16 admission tiers that route through
//! `DistanceStream<DistanceFn>`, ADS is fundamentally per-vertex with
//! sink-side comparison at every chunk boundary — the DistanceFn trait
//! has no hook for that mid-loop bail-out, so this tier prefetches
//! manually and walks the id list directly.
//!
//! ## Calling convention
//!
//! **Callers must rotate the dataset and the query with the same
//! orthogonal matrix** (the ADSampling rotator) before opening a
//! session. L2 is rotation-invariant so the graph topology is
//! unaffected. See [`benchmark::runner::staged_diskann_ads_runner`]
//! for the build-side rotation.

use super::{AdmissionSession, AdmissionStage};
use crate::model::Neighbor;
use diskann::model::InmemDataset;
use vector::FullPrecisionDistance;

/// ADSampling admission stage. Wraps the rotated f32 dataset + the ε
/// constant that controls the scaled-partial-sum confidence test.
pub struct AdsF32Admission<'a, const N: usize> {
    pub dataset: &'a InmemDataset<f32, N>,
    pub ads_epsilon: f32,
}

impl<'a, const N: usize> AdsF32Admission<'a, N> {
    #[inline]
    pub fn new(dataset: &'a InmemDataset<f32, N>, ads_epsilon: f32) -> Self {
        Self {
            dataset,
            ads_epsilon,
        }
    }
}

impl<'a, const N: usize> AdmissionStage<N> for AdsF32Admission<'a, N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    #[inline]
    fn open<'b>(&'b self, q: &[f32; N]) -> Box<dyn AdmissionSession + 'b> {
        Box::new(AdsF32Session {
            dataset: self.dataset,
            query: *q,
            ads_epsilon: self.ads_epsilon,
        })
    }
}

/// Per-query session: copy of the (rotated) query + the ε constant.
/// The dataset reference stays borrowed; `entry_distance` and
/// `admit_stream` read vertices through `get_vertex_unchecked`.
pub struct AdsF32Session<'a, const N: usize> {
    dataset: &'a InmemDataset<f32, N>,
    query: [f32; N],
    ads_epsilon: f32,
}

impl<'a, const N: usize> AdmissionSession for AdsF32Session<'a, N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    #[inline]
    fn entry_distance(&self, vid: u32) -> f32 {
        // Entry hop: PQ is empty so the cutoff is effectively MAX and
        // early-abort can't fire. Skip the ADS kernel and use plain
        // L2 to avoid the per-chunk threshold check overhead.
        unsafe {
            let v = self.dataset.get_vertex_unchecked(vid);
            vector::distance_l2_vector_f32::<N>(&self.query, v)
        }
    }

    #[inline]
    unsafe fn admit_stream(
        &self,
        id_scratch: &[u32],
        out: *mut Neighbor,
        cutoff: f32,
        _lookahead_lines: usize,
    ) -> usize {
        let n = id_scratch.len();
        if n == 0 {
            return 0;
        }
        // Manual per-vertex prefetch — DistanceStream's queue-driven
        // prefetch can't be reused because the ADS kernel mixes
        // partial-sum accumulation with mid-loop bail-out at every
        // chunk boundary, a control-flow shape the chunk-step
        // DistanceFn trait doesn't expose.
        self.dataset.prefetch_vector(id_scratch[0]);

        let mut w = 0usize;
        for m in 0..n {
            if m + 1 < n {
                self.dataset.prefetch_vector(id_scratch[m + 1]);
            }
            let id = id_scratch[m];
            let v = unsafe { self.dataset.get_vertex_unchecked(id) };
            let dist = <[f32; N] as FullPrecisionDistance<f32, N>>::distance_compare_adsampling(
                &self.query,
                v,
                cutoff,
                self.ads_epsilon,
            );
            unsafe {
                out.add(w).write(Neighbor::new(id, dist));
            }
            // Branch-free admit: ADS-not-aborted AND beats the cutoff.
            // `dist < 0` flags an early-abort; `dist ≥ cutoff` flags a
            // "ran to completion but lost the race". Both fail this
            // gate so the slot is recycled on the next iteration.
            let keep = (dist >= 0.0) & (dist < cutoff);
            w += keep as usize;
        }
        w
    }
}
