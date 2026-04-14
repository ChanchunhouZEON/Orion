/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet};

use vector::FullPrecisionDistance;

use super::InmemIndex;
use crate::common::{ANNError, ANNResult};
use crate::model::vertex::{DIM_32, DIM_100};
use crate::model::{
    IndexConfiguration,
    vertex::{DIM_104, DIM_128, DIM_256, DIM_784, DIM_960},
};

/// ANN inmem-index abstraction for custom <T, N>
pub trait ANNInmemIndex<T>: Sync + Send
where
    T: Default + Copy + Sync + Send + Into<f32> + 'static,
{
    /// Downcast to concrete type for field extraction (e.g. taking InmemDataset).
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Build index
    fn build(&mut self, filename: &str, num_points_to_load: usize) -> ANNResult<()>;

    /// Save index
    fn save(&mut self, filename: &str) -> ANNResult<()>;

    /// Load index
    fn load(&mut self, filename: &str, expected_num_points: usize) -> ANNResult<()>;

    /// Insert index
    fn insert(&mut self, filename: &str, num_points_to_insert: usize) -> ANNResult<()>;

    /// Search the index for K nearest neighbors of query using given L value
    fn search(
        &self,
        query: &[T],
        k_value: usize,
        l_value: u32,
        indices: &mut [u32],
    ) -> ANNResult<u32>;

    /// Soft deletes the nodes with the ids in the given array.
    fn soft_delete(
        &mut self,
        vertex_ids_to_delete: Vec<u32>,
        num_points_to_delete: usize,
    ) -> ANNResult<()>;

    /// Save graph in mmap-compatible format for zero-copy search.
    fn save_graph_mmap(&self, path: &str) -> ANNResult<()>;

    /// Save graph with vector data in mmap-compatible format.
    /// Enables mmap search to read vectors directly from the file.
    fn save_graph_mmap_with_vectors(&self, path: &str) -> ANNResult<()>;

    /// Get the start/entry node ID.
    fn start_node(&self) -> u32;

    /// Extract candidate sets (if computed during build).
    #[cfg(feature = "staged_diskann")]
    fn extract_candidate_sets(&mut self) -> Option<Vec<HashSet<u32>>>;

    /// Sort neighbors by distance in-place, enrich slab, extract
    /// InMemoryGraph + candidate_sets. No CsrGraph intermediate.
    #[cfg(feature = "staged_diskann")]
    fn extract_graph_and_candidates(
        &mut self,
        _key_neighbor_count: usize,
    ) -> ANNResult<(crate::model::InMemoryGraph, Vec<Vec<u32>>)> {
        Err(ANNError::log_index_error("not implemented".into()))
    }

    /// Extract graph + candidates with explicit enrich control.
    #[cfg(feature = "staged_diskann")]
    fn extract_graph_and_candidates_ex(
        &mut self,
        _key_neighbor_count: usize,
        _enrich: bool,
    ) -> ANNResult<(crate::model::InMemoryGraph, Vec<Vec<u32>>)> {
        Err(ANNError::log_index_error("not implemented".into()))
    }

    /// Free the candidate anchor slab.
    #[cfg(feature = "staged_diskann")]
    fn drop_candidate_slab(&mut self) {}

    /// Re-run single-pass prune on all nodes to enrich the candidate slab.
    #[cfg(feature = "staged_diskann")]
    fn enrich_candidate_slab(&self) -> ANNResult<()> {
        Ok(())
    }

    /// Extract the final graph as a HashMap for external consumers.
    fn extract_graph(&self) -> HashMap<u32, Vec<u32>> {
        HashMap::new()
    }

    /// Move the final graph out of the index directly, avoiding the HashMap round-trip.
    ///
    /// Default falls back to the HashMap path for implementations that don't expose
    /// `final_graph` directly.  `InmemIndex` overrides this with a zero-alloc
    /// `std::mem::replace`.
    fn extract_final_graph(
        &mut self,
        num_points: usize,
        max_degree: u32,
    ) -> crate::model::InMemoryGraph {
        use crate::model::InMemoryGraph;
        let map = self.extract_graph();
        InMemoryGraph::from_hashmap(&map, num_points, max_degree)
    }

    /// Get number of active points.
    fn num_active_points(&self) -> usize {
        0
    }

    /// Build the index from in-memory data instead of a file.
    fn build_from_data(&mut self, _data: &[T], _num_points: usize) -> ANNResult<()> {
        Err(ANNError::log_index_error(
            "build_from_data not supported".to_string(),
        ))
    }
}

/// Create Index<T, N> based on configuration
pub fn create_inmem_index<T>(config: IndexConfiguration) -> ANNResult<Box<dyn ANNInmemIndex<T>>>
where
    T: Default + Copy + Sync + Send + Into<f32> + 'static,
    [T; DIM_32]: FullPrecisionDistance<T, DIM_32>,
    [T; DIM_100]: FullPrecisionDistance<T, DIM_100>,
    [T; DIM_104]: FullPrecisionDistance<T, DIM_104>,
    [T; DIM_128]: FullPrecisionDistance<T, DIM_128>,
    [T; DIM_256]: FullPrecisionDistance<T, DIM_256>,
    [T; DIM_784]: FullPrecisionDistance<T, DIM_784>,
    [T; DIM_960]: FullPrecisionDistance<T, DIM_960>,
{
    match config.aligned_dim {
        DIM_32 => {
            let index = Box::new(InmemIndex::<T, DIM_32>::new(config)?);
            Ok(index as Box<dyn ANNInmemIndex<T>>)
        }
        DIM_100 => {
            let index = Box::new(InmemIndex::<T, DIM_100>::new(config)?);
            Ok(index as Box<dyn ANNInmemIndex<T>>)
        }
        DIM_104 => {
            let index = Box::new(InmemIndex::<T, DIM_104>::new(config)?);
            Ok(index as Box<dyn ANNInmemIndex<T>>)
        }
        DIM_128 => {
            let index = Box::new(InmemIndex::<T, DIM_128>::new(config)?);
            Ok(index as Box<dyn ANNInmemIndex<T>>)
        }
        DIM_256 => {
            let index = Box::new(InmemIndex::<T, DIM_256>::new(config)?);
            Ok(index as Box<dyn ANNInmemIndex<T>>)
        }
        DIM_784 => {
            let index = Box::new(InmemIndex::<T, DIM_784>::new(config)?);
            Ok(index as Box<dyn ANNInmemIndex<T>>)
        }
        DIM_960 => {
            let index = Box::new(InmemIndex::<T, DIM_960>::new(config)?);
            Ok(index as Box<dyn ANNInmemIndex<T>>)
        }
        _ => Err(ANNError::log_index_error(format!(
            "Invalid dimension: {}",
            config.aligned_dim
        ))),
    }
}
