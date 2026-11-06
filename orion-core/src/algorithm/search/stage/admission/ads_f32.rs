/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! ADSampling f32 admission tier.
//!
//! Uses the scaled-partial-sum L2 kernel from [`orion_ann_vector`]
//! (`distance_compare_adsampling`) — chunked early-abort against the
//! per-hop PQ cutoff. The kernel returns `-1.0` when the partial sum
//! crosses the confidence band before completion, and the L2² sum
//! otherwise. cmov-compact admits on `dist ≥ 0 && dist < cutoff`.
//!
//! Unlike the i8 / i16 admission tiers that route through
//! `DistanceStream<DistanceKernelBackend>`, ADS is fundamentally per-vertex with
//! sink-side comparison at every chunk boundary — the DistanceKernelBackend trait
//! has no hook for that mid-loop bail-out, so this tier prefetches
//! manually and walks the id list directly.
//!
//! ## Calling convention
//!
//! **Callers must rotate the dataset and the query with the same
//! orthogonal matrix** (the ADSampling rotator) before opening a
//! session. L2 is rotation-invariant so the graph topology is
//! unaffected. See `benchmark::runner::orion_ads_runner`
//! for the build-side rotation.

use super::{AdmissionSession, AdmissionStage};
use crate::model::Neighbor;
use orion_ann_diskann::model::InmemDataset;
use orion_ann_vector::FullPrecisionDistance;

/// ADSampling admission stage. Wraps the rotated f32 dataset + the ε
/// constant that controls the scaled-partial-sum confidence test.
pub struct AdsF32Admission<'a, const N: usize> {
    dataset: &'a InmemDataset<f32, N>,
    ads_epsilon: f32,
}

impl<'a, const N: usize> AdsF32Admission<'a, N> {
    #[inline]
    pub fn new(dataset: &'a InmemDataset<f32, N>, ads_epsilon: f32) -> Self {
        assert!(N > 0, "stage dimension must be positive");
        let logical_len = dataset
            .num_active_pts
            .checked_mul(N)
            .expect("dataset span overflow");
        assert!(
            logical_len <= dataset.data.len(),
            "dataset allocation shorter than active rows"
        );
        Self {
            dataset,
            ads_epsilon,
        }
    }
}

// SAFETY: shape describes the borrowed base; sessions own checked query storage
// and cannot outlive that base.
unsafe impl<'a, const N: usize> AdmissionStage for AdsF32Admission<'a, N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    fn precision(
        &self,
    ) -> Option<crate::model::dataset::quantized_dataset::capabilities::Precision> {
        use crate::model::dataset::quantized_dataset::capabilities::Precision as P;
        Some(P::Exact)
    }

    fn shape(&self) -> Option<(usize, usize)> {
        Some((N, self.dataset.num_active_pts))
    }
    #[inline]
    fn open<'b>(&'b self, q: &[f32]) -> Box<dyn AdmissionSession + 'b> {
        assert_eq!(q.len(), N);
        let q: &[f32; N] = q.try_into().expect("query dimension mismatch");
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
struct AdsF32Session<'a, const N: usize> {
    dataset: &'a InmemDataset<f32, N>,
    query: [f32; N],
    ads_epsilon: f32,
}

// SAFETY: streams retain input IDs, write only within the input-sized output
// region, and return the initialized compacted prefix length.
unsafe impl<'a, const N: usize> AdmissionSession for AdsF32Session<'a, N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    #[inline]
    fn entry_distance(&self, vid: u32) -> f32 {
        assert!((vid as usize) < self.dataset.num_active_pts);
        // Entry hop: PQ is empty so the cutoff is effectively MAX and
        // early-abort can't fire. Skip the ADS kernel and use plain
        // L2 to avoid the per-chunk threshold check overhead.
        unsafe {
            let v = self.dataset.get_vertex_unchecked(vid);
            orion_ann_vector::distance_l2_vector_f32::<N>(&self.query, v)
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
        // DistanceKernelBackend trait doesn't expose.
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

/// Runtime-dimensional ADS adapter. The caller supplies the same rotated base
/// and query as for the typed adapter; SIMD padding is excluded from sampling.
pub struct DynamicAdsF32Admission<'a> {
    dataset: &'a orion_ann_diskann::model::PaddedDataset<f32>,
    epsilon: f32,
}
impl<'a> DynamicAdsF32Admission<'a> {
    pub fn new(dataset: &'a orion_ann_diskann::model::PaddedDataset<f32>, epsilon: f32) -> Self {
        Self { dataset, epsilon }
    }
}
// SAFETY: shape describes the borrowed base; sessions own checked query storage
// and cannot outlive that base.
unsafe impl AdmissionStage for DynamicAdsF32Admission<'_> {
    fn precision(
        &self,
    ) -> Option<crate::model::dataset::quantized_dataset::capabilities::Precision> {
        use crate::model::dataset::quantized_dataset::capabilities::Precision as P;
        Some(P::Exact)
    }

    fn shape(&self) -> Option<(usize, usize)> {
        Some((self.dataset.layout().logical_dim(), self.dataset.len()))
    }
    fn open<'b>(&'b self, q: &[f32]) -> Box<dyn AdmissionSession + 'b> {
        assert_eq!(q.len(), self.dataset.layout().logical_dim());
        Box::new(DynamicAdsSession {
            dataset: self.dataset,
            query: q.to_vec(),
            epsilon: self.epsilon,
        })
    }
}
struct DynamicAdsSession<'a> {
    dataset: &'a orion_ann_diskann::model::PaddedDataset<f32>,
    query: Vec<f32>,
    epsilon: f32,
}
// SAFETY: streams retain input IDs, write only within the input-sized output
// region, and return the initialized compacted prefix length.
unsafe impl AdmissionSession for DynamicAdsSession<'_> {
    fn entry_distance(&self, id: u32) -> f32 {
        orion_ann_vector::distance_l2_slice_f32(&self.query, self.dataset.row(id as usize).unwrap())
    }
    unsafe fn admit_stream(&self, ids: &[u32], out: *mut Neighbor, cutoff: f32, _: usize) -> usize {
        let prefetch = |id: u32| {
            let row = self.dataset.row(id as usize).unwrap();
            orion_ann_vector::prefetch_vector(row);
        };
        if let Some(&id) = ids.first() {
            prefetch(id);
        }
        let mut count = 0;
        for (i, &id) in ids.iter().enumerate() {
            if let Some(&next) = ids.get(i + 1) {
                prefetch(next);
            }
            let distance = orion_ann_vector::distance_l2_adsampling_slice_f32(
                &self.query,
                self.dataset.row(id as usize).unwrap(),
                cutoff,
                self.epsilon,
            );
            unsafe {
                out.add(count).write(Neighbor::new(id, distance));
            }
            count += ((distance >= 0.) & (distance < cutoff)) as usize;
        }
        count
    }
}

// Keep diagnostics bounded: do not format vector payloads or working buffers.
impl<'a, const N: usize> std::fmt::Debug for AdsF32Admission<'a, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdsF32Admission")
            .field("dimension", &(N))
            .field("points", &self.dataset.num_active_pts)
            .finish_non_exhaustive()
    }
}

impl<'a, const N: usize> std::fmt::Debug for AdsF32Session<'a, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdsF32Session")
            .field("dimension", &(N))
            .field("points", &self.dataset.num_active_pts)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for DynamicAdsF32Admission<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicAdsF32Admission")
            .field("dimension", &self.dataset.layout().logical_dim())
            .field("points", &self.dataset.len())
            .finish_non_exhaustive()
    }
}
