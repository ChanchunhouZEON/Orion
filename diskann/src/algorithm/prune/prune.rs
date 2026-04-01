/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
#[allow(unused_imports)]
use hashbrown::HashSet;
use vector::{FullPrecisionDistance, Metric};

use crate::common::{ANNError, ANNResult};
use crate::index::InmemIndex;
use crate::model::Neighbor;
use crate::model::graph::AdjacencyList;
use crate::model::neighbor::SortedNeighborVector;
use crate::model::scratch::InMemQueryScratch;

impl<T, const N: usize> InmemIndex<T, N>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    #[allow(clippy::too_many_arguments)]
    fn occlude_list(
        &self,
        location: u32,
        pool: &mut SortedNeighborVector,
        alpha: f32,
        degree: u32,
        max_candidate_size: usize,
        result: &mut AdjacencyList,
        scratch: &mut InMemQueryScratch<T, N>,
        delete_set_ptr: Option<&HashSet<u32>>,
    ) -> ANNResult<()> {
        if pool.is_empty() {
            return Ok(());
        }

        if !result.is_empty() {
            return Err(ANNError::log_index_error(
                "result is not empty.".to_string(),
            ));
        }

        if pool.len() > max_candidate_size {
            pool.truncate(max_candidate_size);
        }

        let occlude_factor = &mut scratch.occlude_factor;
        occlude_factor.clear();
        occlude_factor.resize(pool.len(), 0.0);

        #[cfg(feature = "staged_diskann")]
        let track_candidates = self.candidate_anchor_sets.is_some();
        // Reuse scratch buffer to avoid per-call allocation.
        // Collects (anchor_id, pruned_id) pairs; flushed to candidate_anchor_sets after the loop.
        #[cfg(feature = "staged_diskann")]
        scratch.candidate_buffer.clear();

        let mut cur_alpha = 1.0;
        while cur_alpha <= alpha && result.len() < degree as usize {
            for (i, neighbor) in pool.iter().enumerate() {
                if result.len() >= degree as usize {
                    break;
                }
                if occlude_factor[i] > cur_alpha {
                    continue;
                }
                occlude_factor[i] = f32::MAX;

                if delete_set_ptr.map_or(true, |delete_set| !delete_set.contains(&neighbor.id))
                    && neighbor.id != location
                {
                    result.push(neighbor.id);
                }

                for (j, neighbor2) in pool.iter().enumerate().skip(i + 1) {
                    if occlude_factor[j] > alpha {
                        continue;
                    }

                    let djk = self.get_distance(neighbor2.id, neighbor.id)?;
                    #[cfg(feature = "staged_diskann")]
                    let old_factor = occlude_factor[j];
                    match self.configuration.dist_metric {
                        Metric::L2 | Metric::Cosine => {
                            occlude_factor[j] = if djk == 0.0 {
                                f32::MAX
                            } else {
                                occlude_factor[j].max(neighbor2.distance / djk)
                            };
                        }
                    }

                    // Record pruned candidate: neighbor2 was occluded by neighbor (anchor)
                    #[cfg(feature = "staged_diskann")]
                    if track_candidates && old_factor <= alpha && occlude_factor[j] > alpha {
                        scratch.candidate_buffer.push((neighbor.id, neighbor2.id));
                    }
                }
            }

            cur_alpha *= 1.2;
        }

        // Flush (anchor, pruned_id) pairs to candidate_anchor_sets.
        // Sort by anchor first so we acquire each anchor's lock only once.
        #[cfg(feature = "staged_diskann")]
        if !scratch.candidate_buffer.is_empty() {
            if let Some(ref cas) = self.candidate_anchor_sets {
                if scratch.candidate_buffer.len() > 1 {
                    scratch.candidate_buffer.sort_unstable_by_key(|&(anchor, _)| anchor);
                }
                let mut i = 0;
                while i < scratch.candidate_buffer.len() {
                    let anchor = scratch.candidate_buffer[i].0;
                    if let Some(slot) = cas.get(anchor as usize) {
                        let mut vec = slot.lock().unwrap();
                        while i < scratch.candidate_buffer.len()
                            && scratch.candidate_buffer[i].0 == anchor
                        {
                            vec.push((location, scratch.candidate_buffer[i].1));
                            i += 1;
                        }
                    } else {
                        while i < scratch.candidate_buffer.len()
                            && scratch.candidate_buffer[i].0 == anchor
                        {
                            i += 1;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn prune_neighbors(
        &self,
        location: u32,
        pool: &mut Vec<Neighbor>,
        pruned_list: &mut AdjacencyList,
        scratch: &mut InMemQueryScratch<T, N>,
    ) -> ANNResult<()> {
        self.robust_prune(
            location,
            pool,
            self.configuration.index_write_parameter.max_degree,
            self.configuration.index_write_parameter.max_occlusion_size,
            self.configuration.index_write_parameter.alpha,
            pruned_list,
            scratch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn robust_prune(
        &self,
        location: u32,
        pool: &mut Vec<Neighbor>,
        range: u32,
        max_candidate_size: u32,
        alpha: f32,
        pruned_list: &mut AdjacencyList,
        scratch: &mut InMemQueryScratch<T, N>,
    ) -> ANNResult<()> {
        if pool.is_empty() {
            pruned_list.clear();
            return Ok(());
        }

        let mut pool = SortedNeighborVector::new(pool);
        pruned_list.clear();

        self.occlude_list(
            location,
            &mut pool,
            alpha,
            range,
            max_candidate_size as usize,
            pruned_list,
            scratch,
            Option::None,
        )?;

        if pruned_list.len() > range as usize {
            return Err(ANNError::log_index_error(format!(
                "pruned_list's len {} is over range {}.",
                pruned_list.len(),
                range
            )));
        }

        if self.configuration.index_write_parameter.saturate_graph && alpha > 1.0f32 {
            for neighbor in pool.iter() {
                if pruned_list.len() >= (range as usize) {
                    break;
                }
                if !pruned_list.contains(&neighbor.id) && neighbor.id != location {
                    pruned_list.push(neighbor.id);
                }
            }
        }

        Ok(())
    }

    pub fn inter_insert(
        &self,
        n: u32,
        pruned_list: &Vec<u32>,
        range: u32,
        scratch: &mut InMemQueryScratch<T, N>,
    ) -> ANNResult<()> {
        let src_pool = pruned_list;

        if src_pool.is_empty() {
            return Err(ANNError::log_index_error("src_pool is empty.".to_string()));
        }

        for &vertex_id in src_pool {
            if (vertex_id as usize)
                >= self.configuration.max_points + self.configuration.num_frozen_pts
            {
                return Err(ANNError::log_index_error(format!(
                    "vertex_id {} is out of valid range of points {}",
                    vertex_id,
                    self.configuration.max_points + self.configuration.num_frozen_pts,
                )));
            }

            let neighbors = self.add_to_neighbors(vertex_id, n, range)?;

            if let Some(copy_of_neighbors) = neighbors {
                let mut dummy_pool = self.get_unique_neighbors(&copy_of_neighbors, vertex_id)?;

                let mut new_out_neighbors =
                    AdjacencyList::for_range(self.configuration.write_range());
                self.prune_neighbors(vertex_id, &mut dummy_pool, &mut new_out_neighbors, scratch)?;

                self.set_neighbors(vertex_id, new_out_neighbors)?;
            }
        }

        Ok(())
    }

    fn add_to_neighbors(
        &self,
        vertex_id: u32,
        node_id: u32,
        range: u32,
    ) -> ANNResult<Option<Vec<u32>>> {
        let mut vertex_guard = self.final_graph.write_vertex_and_neighbors(vertex_id)?;

        Ok(vertex_guard.add_to_neighbors(node_id, range))
    }

    fn set_neighbors(&self, vertex_id: u32, new_out_neighbors: AdjacencyList) -> ANNResult<()> {
        let mut vertex_guard = self.final_graph.write_vertex_and_neighbors(vertex_id)?;

        vertex_guard.set_neighbors(new_out_neighbors);
        Ok(())
    }
}
