//! Full-precision reranking of padded rows with the existing distance stream.
use super::RerankStage;
use crate::{
    algorithm::search::{NDC_F32, utils::dstream_la_truth},
    model::{Neighbor, scratch::InMemSearchScratch},
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
        assert_eq!(dataset.layout().stride_bytes() % K::CHUNK_BYTES, 0);
        assert!(dataset.layout().logical_dim() <= K::MAX_COORDINATES);
        Self {
            dataset,
            kernel: PhantomData,
        }
    }
}
impl<K: DistanceKernelBackend<Storage = f32> + Send + Sync> RerankStage
    for DynamicF32Rerank<'_, K>
{
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
            .extend(scratch.pq.neighbors().iter().take(n).map(|v| v.id));
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
        scored.into_iter().take(k).map(|(id, _)| id).collect()
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
