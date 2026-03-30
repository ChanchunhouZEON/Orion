/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use vector::Metric;

use super::index_write_parameters::IndexWriteParameters;

#[derive(Debug, Clone)]
pub struct IndexConfiguration {
    pub index_write_parameter: IndexWriteParameters,
    pub dist_metric: Metric,
    pub dim: usize,
    pub aligned_dim: usize,
    pub max_points: usize,
    pub num_frozen_pts: usize,
    pub use_pq_dist: bool,
    pub num_pq_chunks: usize,
    pub use_opq: bool,
    pub growth_potential: f32,
}

impl IndexConfiguration {
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

    pub fn write_range(&self) -> usize {
        self.index_write_parameter.max_degree as usize
    }
}
