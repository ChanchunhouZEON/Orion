/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::cmp::max;
use std::mem;

use hashbrown::HashSet;

use crate::common::{ANNError, ANNResult, AlignedBoxWithSlice};
use crate::model::configuration::index_write_parameters::IndexWriteParameters;
use crate::model::{Neighbor, NeighborPriorityQueue, PQScratch};

use super::Scratch;

pub const GRAPH_SLACK_FACTOR: f64 = 1.3_f64;
pub const MAX_POINTS_FOR_USING_BITSET: usize = 100000;
pub const MAX_GRAPH_DEGREE: usize = 512;
pub const MAX_N_CMPS: usize = 16384;
pub const SECTOR_LEN: usize = 4096;
pub const MAX_N_SECTOR_READS: usize = 128;
pub const QUERY_ALIGNMENT_OF_T_SIZE: usize = 16;

#[derive(Debug)]
pub struct InMemQueryScratch<T, const N: usize> {
    pub candidate_size: u32,
    pub max_degree: u32,
    pub max_occlusion_size: u32,
    pub query: AlignedBoxWithSlice<T>,
    pub best_candidates: NeighborPriorityQueue,
    pub occlude_factor: Vec<f32>,
    pub id_scratch: Vec<u32>,
    pub dist_scratch: Vec<f32>,
    pub pq_scratch: Option<Box<PQScratch>>,
    pub expanded_nodes_set: HashSet<u32>,
    pub expanded_neighbors_vector: Vec<Neighbor>,
    pub occlude_list_output: Vec<u32>,
    pub node_visited_robinset: HashSet<u32>,

    /// Local buffer for (anchor_id, pruned_id) pairs accumulated in `occlude_list`.
    /// Reused across calls to avoid per-call HashMap allocation.
    #[cfg(feature = "orion")]
    pub candidate_buffer: Vec<(u32, u32)>,
}

impl<T: Default + Copy, const N: usize> InMemQueryScratch<T, N> {
    pub fn new(
        search_candidate_size: u32,
        index_write_parameter: &IndexWriteParameters,
        init_pq_scratch: bool,
    ) -> ANNResult<Self> {
        let indexing_candidate_size = index_write_parameter.search_list_size;
        let max_degree = index_write_parameter.max_degree;
        let max_occlusion_size = index_write_parameter.max_occlusion_size;

        if search_candidate_size == 0 || indexing_candidate_size == 0 || max_degree == 0 || N == 0 {
            return Err(ANNError::log_index_error(format!(
                "In InMemQueryScratch, one of search_candidate_size = {}, indexing_candidate_size = {}, dim = {} or max_degree = {} is zero.",
                search_candidate_size, indexing_candidate_size, N, max_degree
            )));
        }

        let query = AlignedBoxWithSlice::new(N, mem::size_of::<T>() * QUERY_ALIGNMENT_OF_T_SIZE)?;
        let pq_scratch = if init_pq_scratch {
            Some(Box::new(PQScratch::new(MAX_GRAPH_DEGREE, N)?))
        } else {
            None
        };

        let occlude_factor = Vec::with_capacity(max_occlusion_size as usize);

        let capacity = (1.5 * GRAPH_SLACK_FACTOR * (max_degree as f64)).ceil() as usize;
        let id_scratch = Vec::with_capacity(capacity);
        let dist_scratch = Vec::with_capacity(capacity);

        let expanded_nodes_set = HashSet::<u32>::new();
        let expanded_neighbors_vector = Vec::<Neighbor>::new();
        let occlude_list_output = Vec::<u32>::new();

        let candidate_size = max(search_candidate_size, indexing_candidate_size);
        let node_visited_robinset = HashSet::<u32>::with_capacity(20 * candidate_size as usize);
        let scratch = Self {
            candidate_size,
            max_degree,
            max_occlusion_size,
            query,
            best_candidates: NeighborPriorityQueue::with_capacity(candidate_size as usize),
            occlude_factor,
            id_scratch,
            dist_scratch,
            pq_scratch,
            expanded_nodes_set,
            expanded_neighbors_vector,
            occlude_list_output,
            node_visited_robinset,
            #[cfg(feature = "orion")]
            candidate_buffer: Vec::with_capacity(max_occlusion_size as usize),
        };

        Ok(scratch)
    }

    pub fn resize_for_new_candidate_size(&mut self, new_candidate_size: u32) {
        if new_candidate_size > self.candidate_size {
            let delta = new_candidate_size - self.candidate_size;
            self.candidate_size = new_candidate_size;
            self.best_candidates.reserve(delta as usize);
            self.node_visited_robinset.reserve((20 * delta) as usize);
        }
    }
}

impl<T: Default + Copy, const N: usize> Scratch for InMemQueryScratch<T, N> {
    fn clear(&mut self) {
        self.best_candidates.clear();
        self.occlude_factor.clear();
        self.node_visited_robinset.clear();
        self.id_scratch.clear();
        self.dist_scratch.clear();
        self.expanded_nodes_set.clear();
        self.expanded_neighbors_vector.clear();
        self.occlude_list_output.clear();
        #[cfg(feature = "orion")]
        self.candidate_buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::configuration::index_write_parameters::IndexWriteParametersBuilder;

    fn make_write_params() -> IndexWriteParameters {
        IndexWriteParametersBuilder::new(10, 32)
            .with_alpha(1.2)
            .build()
    }

    #[test]
    fn test_new_scratch() {
        let params = make_write_params();
        let scratch = InMemQueryScratch::<f32, 8>::new(10, &params, false).unwrap();
        assert_eq!(scratch.candidate_size, 10);
        assert_eq!(scratch.max_degree, 32);
    }

    #[test]
    fn test_clear_resets_all_fields() {
        let params = make_write_params();
        let mut scratch = InMemQueryScratch::<f32, 8>::new(10, &params, false).unwrap();

        scratch.id_scratch.push(1);
        scratch.dist_scratch.push(1.0);
        scratch.expanded_nodes_set.insert(1);
        scratch
            .expanded_neighbors_vector
            .push(Neighbor::new(0, 1.0));
        scratch.occlude_list_output.push(1);
        scratch.node_visited_robinset.insert(1);

        scratch.clear();

        assert!(scratch.id_scratch.is_empty());
        assert!(scratch.dist_scratch.is_empty());
        assert!(scratch.expanded_nodes_set.is_empty());
        assert!(scratch.expanded_neighbors_vector.is_empty());
        assert!(scratch.occlude_list_output.is_empty());
        assert!(scratch.node_visited_robinset.is_empty());
    }

    #[test]
    fn test_zero_params_returns_error() {
        let params = IndexWriteParametersBuilder::new(0, 0).build();
        let result = InMemQueryScratch::<f32, 8>::new(0, &params, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_resize_for_larger_candidate_size() {
        let params = make_write_params();
        let mut scratch = InMemQueryScratch::<f32, 8>::new(10, &params, false).unwrap();
        assert_eq!(scratch.candidate_size, 10);

        scratch.resize_for_new_candidate_size(20);
        assert_eq!(scratch.candidate_size, 20);
    }

    #[test]
    fn test_resize_no_shrink() {
        let params = make_write_params();
        let mut scratch = InMemQueryScratch::<f32, 8>::new(10, &params, false).unwrap();
        scratch.resize_for_new_candidate_size(5); // smaller, should be no-op
        assert_eq!(scratch.candidate_size, 10);
    }
}
