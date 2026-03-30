/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::common::{ANNError, ANNResult};
use crate::index::InmemIndex;
use crate::model::{Neighbor, Vertex, scratch::InMemQueryScratch};
use hashbrown::hash_set::Entry::*;
use vector::FullPrecisionDistance;

impl<T, const N: usize> InmemIndex<T, N>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    pub fn search_with_l_override(
        &self,
        query: &Vertex<T, N>,
        scratch: &mut InMemQueryScratch<T, N>,
        search_list_size: usize,
    ) -> ANNResult<u32> {
        let init_ids = self.get_init_ids()?;
        self.init_graph_for_point(query, init_ids, scratch)?;
        scratch.best_candidates.set_capacity(search_list_size);
        let (_, cmp) = self.greedy_search(query, scratch)?;

        Ok(cmp)
    }

    pub fn search_for_point(
        &self,
        query: &Vertex<T, N>,
        scratch: &mut InMemQueryScratch<T, N>,
    ) -> ANNResult<Vec<Neighbor>> {
        let init_ids = self.get_init_ids()?;
        self.init_graph_for_point(query, init_ids, scratch)?;
        let (mut visited_nodes, _) = self.greedy_search(query, scratch)?;

        visited_nodes.retain(|&element| element.id != query.vertex_id());
        Ok(visited_nodes)
    }

    fn get_init_ids(&self) -> ANNResult<Vec<u32>> {
        let mut init_ids = Vec::with_capacity(1 + self.configuration.num_frozen_pts);
        init_ids.push(self.start);

        for frozen in self.configuration.max_points
            ..(self.configuration.max_points + self.configuration.num_frozen_pts)
        {
            let frozen_u32 = frozen.try_into()?;
            if frozen_u32 != self.start {
                init_ids.push(frozen_u32);
            }
        }

        Ok(init_ids)
    }

    fn init_graph_for_point(
        &self,
        query: &Vertex<T, N>,
        init_ids: Vec<u32>,
        scratch: &mut InMemQueryScratch<T, N>,
    ) -> ANNResult<()> {
        scratch
            .best_candidates
            .reserve(self.configuration.index_write_parameter.search_list_size as usize);
        scratch.query.memcpy(query.vector())?;

        if !scratch.id_scratch.is_empty() {
            return Err(ANNError::log_index_error(
                "id_scratch is not empty.".to_string(),
            ));
        }

        let query_vertex = Vertex::<T, N>::try_from((&scratch.query[..], query.vertex_id()))
            .map_err(|err| {
                ANNError::log_index_error(format!(
                    "TryFromSliceError: failed to get Vertex for query, err={}",
                    err
                ))
            })?;

        for id in init_ids {
            if (id as usize) >= self.configuration.max_points + self.configuration.num_frozen_pts {
                return Err(ANNError::log_index_error(format!(
                    "vertex_id {} is out of valid range of points {}",
                    id,
                    self.configuration.max_points + self.configuration.num_frozen_pts
                )));
            }

            if let Vacant(entry) = scratch.node_visited_robinset.entry(id) {
                entry.insert();

                let vertex = self.dataset.get_vertex(id)?;

                let distance = vertex.compare(&query_vertex, self.configuration.dist_metric);
                let neighbor = Neighbor::new(id, distance);
                scratch.best_candidates.insert(neighbor);
            }
        }

        Ok(())
    }

    fn greedy_search(
        &self,
        query: &Vertex<T, N>,
        scratch: &mut InMemQueryScratch<T, N>,
    ) -> ANNResult<(Vec<Neighbor>, u32)> {
        let mut visited_nodes =
            Vec::with_capacity((3 * scratch.candidate_size + scratch.max_degree) as usize);

        let mut cmps: u32 = 0;

        let query_vertex = Vertex::<T, N>::try_from((&scratch.query[..], query.vertex_id()))
            .map_err(|err| {
                ANNError::log_index_error(format!(
                    "TryFromSliceError: failed to get Vertex for query, err={}",
                    err
                ))
            })?;

        while scratch.best_candidates.has_notvisited_node() {
            let closest_node = scratch.best_candidates.closest_notvisited();

            visited_nodes.push(closest_node);

            scratch.id_scratch.clear();

            let max_vertex_id = self.configuration.max_points + self.configuration.num_frozen_pts;

            for id in self
                .final_graph
                .read_vertex_and_neighbors(closest_node.id)?
                .get_neighbors()
            {
                let current_vertex_id = *id;
                debug_assert!(
                    (current_vertex_id as usize) < max_vertex_id,
                    "current_vertex_id {} is out of valid range of points {}",
                    current_vertex_id,
                    max_vertex_id
                );
                if current_vertex_id as usize >= max_vertex_id {
                    continue;
                }

                if scratch.node_visited_robinset.insert(current_vertex_id) {
                    scratch.id_scratch.push(current_vertex_id);
                }
            }

            let len = scratch.id_scratch.len();
            for (m, &id) in scratch.id_scratch.iter().enumerate() {
                if m + 1 < len {
                    let next_node = unsafe { *scratch.id_scratch.get_unchecked(m + 1) };
                    self.dataset.prefetch_vector(next_node);
                }

                let vertex = self.dataset.get_vertex(id)?;
                let distance = query_vertex.compare(&vertex, self.configuration.dist_metric);

                scratch.best_candidates.insert(Neighbor::new(id, distance));
            }

            cmps += len as u32;
        }

        Ok((visited_nodes, cmps))
    }
}
