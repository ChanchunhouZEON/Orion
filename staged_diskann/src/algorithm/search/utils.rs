/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use crate::StagedDiskANN;
use diskann::common::{ANNError, ANNResult};
use diskann::model::FixedChunkPQTable;
use std::sync::Arc;
use vector::FullPrecisionDistance;

impl<const N: usize> StagedDiskANN<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Compute PQ distance for a point given its ID and pre-computed chunk distances.
    /// SAFETY: Should be invoked only when pq is activated.
    #[inline]
    #[allow(dead_code)]
    pub(super) fn pq_distance(&self, point_id: u32, pq_dists: &[f32]) -> ANNResult<f32> {
        let (pq, pq_codes, num_pq_chunks) = self.get_unwrapped_pq_component()?;

        let idx = point_id as usize;
        let code_start = idx * num_pq_chunks;
        let code = &pq_codes[code_start..code_start + num_pq_chunks];
        Ok(pq.adc_distance(code, pq_dists))
    }

    pub(super) fn get_unwrapped_pq_component(
        &self,
    ) -> ANNResult<(&Arc<FixedChunkPQTable>, &Vec<u8>, usize)> {
        let pq = self.pq.as_ref().ok_or_else(|| {
            ANNError::log_pq_error("Fixed Chunk PQ Table is None for now".to_string())
        })?;
        let pq_codes = self
            .pq_codes
            .as_ref()
            .ok_or_else(|| ANNError::log_pq_error("PQ codes is None for now".to_string()))?;
        let num_pq_chunks = self.num_pq_chunks.ok_or_else(|| {
            ANNError::log_pq_error("Number of pq chunks is None for now".to_string())
        })?;

        Ok((pq, pq_codes, num_pq_chunks))
    }
}

/// Detailed profiling breakdown for a single search query.
#[derive(Debug, Clone)]
pub struct SearchProfile {
    pub total_us: f64,
    pub adc_table_us: f64,
    pub phase1_us: f64,
    pub phase2_us: f64,
    pub phase2_graph_read_us: f64,
    pub phase2_prefetch_us: f64,
    pub phase2_adc_us: f64,
    pub phase2_async_adc_us: f64,
    pub rerank_us: f64,
    pub phase1_iters: u32,
    pub phase2_iters: u32,
    pub visited_count: u32,
    pub cache_hits: u32,
    pub async_batches: u32,
}
