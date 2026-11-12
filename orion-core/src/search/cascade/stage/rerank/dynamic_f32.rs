/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Full-precision reranking of padded rows with the existing distance stream.
use super::RerankStage;
use crate::{
    model::{Neighbor, scratch::InMemSearchScratch},
    search::{NDC_F32, traversal::tuning::dstream_la_truth},
};
use orion_ann_diskann::model::PaddedDataset;
use orion_ann_vector::{DistanceKernelBackend, L2F32Distance};
use std::marker::PhantomData;

pub struct DynamicF32Rerank<'a, K: DistanceKernelBackend<Storage = f32> = L2F32Distance> {
    dataset: &'a PaddedDataset<f32>,
    kernel: PhantomData<K>,
}
impl<'a, K: DistanceKernelBackend<Storage = f32>> DynamicF32Rerank<'a, K> {
    pub fn new(dataset: &'a PaddedDataset<f32>) -> Self {
        assert_eq!(
            dataset.layout().stride_bytes()
                % K::stream_block_bytes(dataset.layout().compute_bytes()),
            0
        );
        assert!(dataset.layout().logical_dim() <= K::MAX_COORDINATES);
        Self {
            dataset,
            kernel: PhantomData,
        }
    }
}
impl<K: DistanceKernelBackend<Storage = f32> + Send + Sync + 'static> RerankStage
    for DynamicF32Rerank<'_, K>
{
    fn exact(
        &self,
    ) -> Option<crate::model::dataset::quantized_dataset::exact::stage::ExactStage<'_>> {
        use crate::{ScoreMetric, model::dataset::quantized_dataset::exact::stage::ExactStage};
        use std::any::TypeId;
        let metric = if TypeId::of::<K>() == TypeId::of::<L2F32Distance>() {
            ScoreMetric::SquaredL2
        } else if TypeId::of::<K>() == TypeId::of::<orion_ann_vector::IpF32Distance>() {
            ScoreMetric::InnerProduct
        } else {
            return None;
        };
        Some(ExactStage::f32(self.dataset, metric).expect("validated rerank layout"))
    }

    fn precision(
        &self,
    ) -> Option<crate::model::dataset::quantized_dataset::capabilities::Precision> {
        use crate::model::dataset::quantized_dataset::capabilities::Precision as P;
        Some(P::Exact)
    }

    fn shape(&self) -> Option<(usize, usize)> {
        Some((self.dataset.layout().logical_dim(), self.dataset.len()))
    }
    fn rerank(
        &self,
        query: &[f32],
        _: &[Neighbor],
        k: usize,
        factor: usize,
        scratch: &mut InMemSearchScratch,
    ) -> Vec<u32> {
        self.ranked(query, k, factor, scratch)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    fn rerank_scored(
        &self,
        query: &[f32],
        k: usize,
        factor: usize,
        scratch: &mut InMemSearchScratch,
        metric: crate::ScoreMetric,
    ) -> (Vec<u32>, Option<Vec<f32>>) {
        use std::any::TypeId;
        let rows = self.ranked(query, k, factor, scratch);
        let compatible = match metric {
            crate::ScoreMetric::SquaredL2 => TypeId::of::<K>() == TypeId::of::<L2F32Distance>(),
            crate::ScoreMetric::InnerProduct => {
                TypeId::of::<K>() == TypeId::of::<orion_ann_vector::IpF32Distance>()
            }
            // Recompute norms to preserve cosine semantics after f32 normalization.
            crate::ScoreMetric::CosineSimilarity => false,
        };
        let scores = compatible.then(|| {
            rows.iter()
                .map(|(_, score)| {
                    if metric == crate::ScoreMetric::InnerProduct {
                        -*score
                    } else {
                        *score
                    }
                })
                .collect()
        });
        (rows.into_iter().map(|(id, _)| id).collect(), scores)
    }
}

impl<K: DistanceKernelBackend<Storage = f32>> DynamicF32Rerank<'_, K> {
    fn ranked(
        &self,
        query: &[f32],
        k: usize,
        factor: usize,
        scratch: &mut InMemSearchScratch,
    ) -> Vec<(u32, f32)> {
        assert_eq!(
            query.len(),
            self.dataset.layout().logical_dim(),
            "query dimension mismatch"
        );
        let n = k.saturating_mul(factor).min(scratch.pq.size());
        if n == 0 {
            return Vec::new();
        }
        // Reuse worker scratch just like the typed reranker. Padding is neutral
        // for both L2 and IP; it never contributes to normalization or scores.
        scratch.q_query_f32_padded.clear();
        scratch.q_query_f32_padded.extend_from_slice(query);
        scratch
            .q_query_f32_padded
            .resize(self.dataset.layout().stride_elements(), 0.);
        scratch.id_scratch.clear();
        scratch
            .id_scratch
            .extend(scratch.pq.neighbors().iter().take(n).map(|v| {
                assert!(
                    (v.id as usize) < self.dataset.len(),
                    "rerank row ID out of bounds"
                );
                v.id
            }));
        let mut scored = Vec::with_capacity(n);
        unsafe {
            orion_ann_vector::DistanceStream::<K>::new(
                self.dataset.as_slice().as_ptr().cast(),
                self.dataset.layout().stride_bytes(),
                self.dataset.layout().stride_bytes(),
                scratch.q_query_f32_padded.as_ptr().cast(),
                &scratch.id_scratch,
                dstream_la_truth(),
            )
            .run(|id, distance| scored.push((id, distance)));
        }
        NDC_F32.add(n as u64);
        scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(k);
        scored
    }
}

// Keep diagnostics bounded: do not format vector payloads or working buffers.
impl<'a, K: DistanceKernelBackend<Storage = f32>> std::fmt::Debug for DynamicF32Rerank<'a, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicF32Rerank")
            .field("dimension", &self.dataset.layout().logical_dim())
            .field("points", &self.dataset.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[should_panic(expected = "rerank row ID out of bounds")]
    fn rejects_invalid_candidate_before_streaming() {
        let layout = orion_ann_vector::PaddedLayout::for_kernel::<L2F32Distance>(3).unwrap();
        let base = PaddedDataset::from_packed(&[0.; 6], layout).unwrap();
        let rerank = DynamicF32Rerank::<L2F32Distance>::new(&base);
        let mut scratch = InMemSearchScratch::new(4);
        scratch.pq.insert(Neighbor::new(2, 0.));
        rerank.rerank(&[0.; 3], &[], 1, 1, &mut scratch);
    }
}
