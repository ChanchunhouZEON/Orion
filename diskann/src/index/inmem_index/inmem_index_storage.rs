/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use byteorder::{LittleEndian, ReadBytesExt};
use num_traits::ToBytes;
#[allow(unused_imports)]
use std::collections::HashSet as StdHashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use vector::FullPrecisionDistance;

use crate::common::{ANNError, ANNResult};
use crate::model::InMemoryGraph;
use crate::model::graph::AdjacencyList;
use crate::utils::{file_exists, save_data_in_base_dimensions};

use super::InmemIndex;
#[cfg(feature = "staged_diskann")]
pub const CANDIDATE_SETS_FILE_HEADER_SIZE: usize = 16;

impl<T, const N: usize> InmemIndex<T, N>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    pub fn load_graph(&mut self, filename: &str, expected_num_points: usize) -> ANNResult<usize> {
        let mut in_file = BufReader::new(File::open(Path::new(filename))?);

        let expected_file_size: usize = in_file.read_u64::<LittleEndian>()? as usize;
        self.max_observed_degree = in_file.read_u32::<LittleEndian>()?;
        self.start = in_file.read_u32::<LittleEndian>()?;
        let file_frozen_pts: usize = in_file.read_u64::<LittleEndian>()? as usize;

        let vamana_metadata_size = 24;

        println!(
            "From graph header, expected_file_size: {}, max_observed_degree: {}, start: {}, file_frozen_pts: {}",
            expected_file_size, self.max_observed_degree, self.start, file_frozen_pts
        );

        if file_frozen_pts != self.configuration.num_frozen_pts {
            if file_frozen_pts == 1 {
                return Err(ANNError::log_index_config_error(
                    "num_frozen_pts".to_string(),
                    "ERROR: When loading index, detected dynamic index, but constructor asks for static index. Exitting.".to_string())
                );
            } else {
                return Err(ANNError::log_index_config_error(
                    "num_frozen_pts".to_string(),
                    "ERROR: When loading index, detected static index, but constructor asks for dynamic index. Exitting.".to_string())
                );
            }
        }

        println!("Loading vamana graph {}...", filename);

        let expected_max_points = expected_num_points - file_frozen_pts;

        if self.configuration.max_points < expected_max_points {
            println!(
                "Number of points in data: {} is greater than max_points: {} Setting max points to: {}",
                expected_max_points, self.configuration.max_points, expected_max_points
            );

            self.configuration.max_points = expected_max_points;
            self.final_graph = InMemoryGraph::new(
                self.configuration.max_points + self.configuration.num_frozen_pts,
                self.configuration.index_write_parameter.max_degree,
            );
        }

        let mut bytes_read = vamana_metadata_size;
        let mut num_edges = 0;
        let mut nodes_read = 0;
        let mut max_observed_degree = 0;

        while bytes_read != expected_file_size {
            let num_nbrs = in_file.read_u32::<LittleEndian>()?;
            max_observed_degree = if num_nbrs > max_observed_degree {
                num_nbrs
            } else {
                max_observed_degree
            };

            if num_nbrs == 0 {
                return Err(ANNError::log_index_error(format!(
                    "ERROR: Point found with no out-neighbors, point# {}",
                    nodes_read
                )));
            }

            num_edges += num_nbrs;
            nodes_read += 1;
            let mut tmp: Vec<u32> = Vec::with_capacity(num_nbrs as usize);
            for _ in 0..num_nbrs {
                tmp.push(in_file.read_u32::<LittleEndian>()?);
            }

            self.final_graph
                .write_vertex_and_neighbors(nodes_read - 1)?
                .set_neighbors(AdjacencyList::from(tmp));
            bytes_read += 4 * (num_nbrs as usize + 1);
        }

        println!(
            "Done. Index has {} nodes and {} out-edges, _start is set to {}",
            nodes_read, num_edges, self.start
        );

        self.max_observed_degree = max_observed_degree;
        Ok(nodes_read as usize)
    }

    pub fn save_graph(&mut self, graph_file: &str) -> ANNResult<u64> {
        let file: File = File::create(graph_file)?;
        let mut out = BufWriter::new(file);

        let file_offset: u64 = 0;
        out.seek(SeekFrom::Start(file_offset))?;
        let mut index_size: u64 = 24;
        let mut max_degree: u32 = 0;

        // #[cfg(feature = "staged_diskann")]
        // let mut candidates_offset: u32 = 0;

        out.write_all(&index_size.to_le_bytes())?;
        out.write_all(&self.max_observed_degree.to_le_bytes())?;

        // #[cfg(feature = "staged_diskann")]
        // out.write_all(&candidates_offset.to_le_bytes())?;

        out.write_all(&self.start.to_le_bytes())?;
        out.write_all(&(self.configuration.num_frozen_pts as u64).to_le_bytes())?;

        for i in 0..self.num_active_pts + self.configuration.num_frozen_pts {
            let idx = i as u32;
            let gk: u32 = self.final_graph.read_vertex_and_neighbors(idx)?.size() as u32;
            out.write_all(&gk.to_le_bytes())?;
            for neighbor in self
                .final_graph
                .read_vertex_and_neighbors(idx)?
                .get_neighbors()
                .iter()
            {
                out.write_all(&neighbor.to_le_bytes())?;
            }
            max_degree =
                if self.final_graph.read_vertex_and_neighbors(idx)?.size() as u32 > max_degree {
                    self.final_graph.read_vertex_and_neighbors(idx)?.size() as u32
                } else {
                    max_degree
                };
            index_size += (std::mem::size_of::<u32>() * (gk as usize + 1)) as u64;
        }

        // Save the candidate sets if `staged_diskann` feature activated
        // #[cfg(feature = "staged_diskann")]
        // {
        //     candidates_offset = index_size as u32;
        //     if self.candidate_sets.is_some() {
        //         let candidate_sets = self.extract_candidate_sets()?;
        //         for i in 0..self.num_active_pts + self.configuration.num_frozen_pts {
        //             let candidate_set_size: u32 = candidate_sets[i].len() as u32;
        //             out.write_all(&candidate_set_size.to_le_bytes())?;
        //             for candidate in &candidate_sets[i] {
        //                 out.write_all(&candidate.to_le_bytes())?;
        //             }
        //
        //             index_size += (std::mem::size_of::<u32>() * (candidate_set_size as usize + 1)) as u64;
        //         }
        //     }
        // }

        out.seek(SeekFrom::Start(file_offset))?;
        out.write_all(&index_size.to_le_bytes())?;
        out.write_all(&max_degree.to_le_bytes())?;

        // #[cfg(feature = "staged_diskann")]
        // out.write_all(&candidates_offset.to_le_bytes())?;

        out.flush()?;
        Ok(index_size)
    }

    pub fn save_data(&mut self, data_file: &str) -> ANNResult<usize> {
        Ok(save_data_in_base_dimensions(
            data_file,
            &mut self.dataset.data,
            self.num_active_pts + self.configuration.num_frozen_pts,
            self.configuration.dim,
            self.configuration.aligned_dim,
            0,
        )?)
    }

    pub fn save_delete_list(&mut self, delete_list_file: &str) -> ANNResult<usize> {
        let mut delete_file_size = 0;
        if let Ok(delete_set) = self.delete_set.read() {
            let delete_set_len = delete_set.len() as u32;

            if delete_set_len != 0 {
                let file: File = File::create(delete_list_file)?;
                let mut writer = BufWriter::new(file);

                writer.write_all(&delete_set_len.to_le_bytes())?;
                delete_file_size += std::mem::size_of::<u32>();

                for &item in delete_set.iter() {
                    writer.write_all(&item.to_be_bytes())?;
                    delete_file_size += std::mem::size_of::<u32>();
                }

                writer.flush()?;
            }
        } else {
            return Err(ANNError::log_lock_poison_error(
                "Poisoned lock on delete set. Can't save deleted list.".to_string(),
            ));
        }

        Ok(delete_file_size)
    }

    pub fn load_delete_list(&mut self, delete_list_file: &str) -> ANNResult<usize> {
        let mut len = 0;

        if file_exists(delete_list_file) {
            let file = File::open(delete_list_file)?;
            let mut reader = BufReader::new(file);

            len = reader.read_u32::<LittleEndian>()? as usize;

            if let Ok(mut delete_set) = self.delete_set.write() {
                for _ in 0..len {
                    let item = reader.read_u32::<LittleEndian>()?;
                    delete_set.insert(item);
                }
            } else {
                return Err(ANNError::log_lock_poison_error(
                    "Poisoned lock on delete set. Can't load deleted list.".to_string(),
                ));
            }
        }

        Ok(len)
    }

    /// Save candidate sets to a binary file.
    ///
    /// Uses `extract_candidate_sets()` to compute the final candidate sets from
    /// the per-anchor pruned-pair slab and graph structure, then serializes them.
    ///
    /// Binary format (all little-endian):
    ///   - file_size:  u64  (total bytes excluding this field itself... actually total data size)
    ///   - num_points: u64
    ///   - Per node:
    ///       num_candidates: u32
    ///       candidate_ids:  [u32; num_candidates]
    #[cfg(feature = "staged_diskann")]
    pub fn save_candidate_sets(&mut self, candidate_sets_file: &str) -> ANNResult<u64> {
        let candidate_sets = self.extract_candidate_sets()?;

        let num_points = candidate_sets.len() as u64;
        let file = File::create(candidate_sets_file)?;
        let mut out = BufWriter::new(file);

        // Write placeholder header; we'll seek back to fill in file_size.
        out.seek(SeekFrom::Start(0))?;
        let mut file_size: u64 = CANDIDATE_SETS_FILE_HEADER_SIZE as u64;
        out.write_all(&file_size.to_le_bytes())?;
        out.write_all(&num_points.to_le_bytes())?;

        let mut total_candidates: u64 = 0;

        for candidates in &candidate_sets {
            let num_candidates = candidates.len() as u32;
            out.write_all(&num_candidates.to_le_bytes())?;

            for &candidate_id in candidates {
                out.write_all(&candidate_id.to_le_bytes())?;
            }

            total_candidates += num_candidates as u64;
            file_size += (std::mem::size_of::<u32>() * (num_candidates as usize + 1)) as u64;
        }

        // Seek back and write the actual file_size.
        out.seek(SeekFrom::Start(0))?;
        out.write_all(&file_size.to_le_bytes())?;
        out.flush()?;

        println!(
            "Saved candidate sets: {} nodes, {} total candidates to {}",
            num_points, total_candidates, candidate_sets_file
        );

        Ok(file_size)
    }

    /// Load candidate sets from a binary file.
    ///
    /// Returns the loaded candidate sets as `Vec<HashSet<u32>>`.
    /// The binary format matches what `save_candidate_sets` produces.
    #[cfg(feature = "staged_diskann")]
    pub fn load_candidate_sets(candidate_sets_file: &str) -> ANNResult<Vec<StdHashSet<u32>>> {
        if !file_exists(candidate_sets_file) {
            return Err(ANNError::log_index_error(format!(
                "Candidate sets file not found: {}",
                candidate_sets_file
            )));
        }

        let file = File::open(candidate_sets_file)?;
        let mut reader = BufReader::new(file);

        let expected_file_size = reader.read_u64::<LittleEndian>()? as usize;
        let num_points = reader.read_u64::<LittleEndian>()? as usize;

        println!(
            "Loading candidate sets from {}: expected_file_size={}, num_points={}",
            candidate_sets_file, expected_file_size, num_points
        );

        let mut candidate_sets: Vec<StdHashSet<u32>> = Vec::with_capacity(num_points);
        let mut bytes_read = CANDIDATE_SETS_FILE_HEADER_SIZE;
        let mut total_candidates: u64 = 0;

        for i in 0..num_points {
            if bytes_read >= expected_file_size {
                return Err(ANNError::log_index_error(format!(
                    "Unexpected end of candidate sets file at node {}",
                    i
                )));
            }

            let num_candidates = reader.read_u32::<LittleEndian>()? as usize;
            let mut candidates = StdHashSet::with_capacity(num_candidates);

            for _ in 0..num_candidates {
                candidates.insert(reader.read_u32::<LittleEndian>()?);
            }

            bytes_read += std::mem::size_of::<u32>() * (num_candidates + 1);
            total_candidates += num_candidates as u64;
            candidate_sets.push(candidates);
        }

        if bytes_read != expected_file_size {
            return Err(ANNError::log_index_error(format!(
                "Candidate sets file size mismatch: read {} bytes, expected {}",
                bytes_read, expected_file_size
            )));
        }

        println!(
            "Done. Loaded candidate sets for {} nodes with {} total candidates",
            num_points, total_candidates
        );

        Ok(candidate_sets)
    }
}
