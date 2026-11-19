/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use orion_ann_vector::Metric;

use super::index_write_parameters::IndexWriteParameters;

/// Combines vector shape, metric, index capacity and construction parameters.
#[derive(Debug, Clone)]
pub struct IndexConfiguration {
    /// Resolved construction and pruning parameters.
    pub index_write_parameter: IndexWriteParameters,

    /// Legacy distance metric used by construction and traversal.
    pub dist_metric: Metric,

    /// Logical coordinates in each input vector.
    pub dim: usize,

    /// Physical coordinate stride including alignment padding.
    pub aligned_dim: usize,

    /// Configured upper bound on stored vector count.
    pub max_points: usize,

    /// Number of reserved frozen vertices.
    pub num_frozen_pts: usize,

    /// Whether legacy distance evaluation uses PQ lookup tables.
    pub use_pq_dist: bool,

    /// Number of independently encoded PQ subspaces.
    pub num_pq_chunks: usize,

    /// Whether legacy PQ preparation requests an orthogonal rotation.
    pub use_opq: bool,

    /// Capacity multiplier reserved for subsequent insertions.
    pub growth_potential: f32,
}

impl IndexConfiguration {
    /// Collects base shape, graph settings and optional PQ policy without allocating index storage.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dist_metric: Metric,
        dim: usize,
        aligned_dim: usize,
        max_points: usize,
        use_pq_dist: bool,
        num_pq_chunks: usize,
        use_opq: bool,
        num_frozen_pts: usize,
        growth_potential: f32,
        index_write_parameter: IndexWriteParameters,
    ) -> Self {
        Self {
            index_write_parameter,
            dist_metric,
            dim,
            aligned_dim,
            max_points,
            num_frozen_pts,
            use_pq_dist,
            num_pq_chunks,
            use_opq,
            growth_potential,
        }
    }

    /// Returns the configured outgoing-neighbor bound used when allocating adjacency lists.
    pub fn write_range(&self) -> usize {
        self.index_write_parameter.max_degree as usize
    }
}
