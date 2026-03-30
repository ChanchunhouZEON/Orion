/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::mem::size_of;

use crate::common::{ANNResult, AlignedBoxWithSlice};

const MAX_PQ_CHUNKS: usize = 512;

#[derive(Debug)]
pub struct PQScratch {
    pub aligned_pqtable_dist_scratch: AlignedBoxWithSlice<f32>,
    pub aligned_dist_scratch: AlignedBoxWithSlice<f32>,
    pub aligned_pq_coord_scratch: AlignedBoxWithSlice<u8>,
    pub rotated_query: AlignedBoxWithSlice<f32>,
    pub aligned_query_float: AlignedBoxWithSlice<f32>,
}

impl PQScratch {
    const ALIGNED_ALLOC_256: usize = 256;

    pub fn new(graph_degree: usize, aligned_dim: usize) -> ANNResult<Self> {
        let aligned_pq_coord_scratch =
            AlignedBoxWithSlice::new(graph_degree * MAX_PQ_CHUNKS, PQScratch::ALIGNED_ALLOC_256)?;
        let aligned_pqtable_dist_scratch =
            AlignedBoxWithSlice::new(256 * MAX_PQ_CHUNKS, PQScratch::ALIGNED_ALLOC_256)?;
        let aligned_dist_scratch =
            AlignedBoxWithSlice::new(graph_degree, PQScratch::ALIGNED_ALLOC_256)?;
        let aligned_query_float = AlignedBoxWithSlice::new(aligned_dim, 8 * size_of::<f32>())?;
        let rotated_query = AlignedBoxWithSlice::new(aligned_dim, 8 * size_of::<f32>())?;

        Ok(Self {
            aligned_pqtable_dist_scratch,
            aligned_dist_scratch,
            aligned_pq_coord_scratch,
            rotated_query,
            aligned_query_float,
        })
    }

    pub fn set<T>(&mut self, dim: usize, query: &[T], norm: f32)
    where
        T: Into<f32> + Copy,
    {
        for (d, item) in query.iter().enumerate().take(dim) {
            let query_val: f32 = (*item).into();
            if (norm - 1.0).abs() > f32::EPSILON {
                self.rotated_query[d] = query_val / norm;
                self.aligned_query_float[d] = query_val / norm;
            } else {
                self.rotated_query[d] = query_val;
                self.aligned_query_float[d] = query_val;
            }
        }
    }
}
