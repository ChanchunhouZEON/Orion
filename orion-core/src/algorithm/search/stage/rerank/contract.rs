/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::{InMemSearchScratch, Neighbor};

/// Final precision pass on the top-(k · rerank_factor) PQ entries.
pub trait RerankStage: Send + Sync {
    /// Logical dimension and row count, for checking recipe/base compatibility.
    fn shape(&self) -> Option<(usize, usize)> {
        None
    }
    /// Score and emit top-k.
    ///
    /// * `query` — the original f32 query (pre-quantization).
    /// * `pq_entries` — top entries from the PQ at end of beam.
    ///   Already in distance-ascending order. Pass the first
    ///   `min(k · rerank_factor, pq_entries.len())` to the rerank
    ///   logic.
    /// * `scratch` — reused buffers for the rerank's id-list staging
    ///   and padded-query allocation.
    fn rerank(
        &self,
        query: &[f32],
        pq_entries: &[Neighbor],
        k: usize,
        rerank_factor: usize,
        scratch: &mut InMemSearchScratch,
    ) -> Vec<u32>;
}
