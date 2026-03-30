/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use crate::common::{ANNError, ANNResult};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Number of PQ centroids per chunk (always 256 for 8-bit codes).
const NUM_PQ_CENTROIDS: usize = 256;

/// Fixed-chunk PQ table that stores centroids in a flat layout compatible
/// with the DiskANN compressed-vector format.
///
/// **Data layout**:
/// - `pq_table`: flat `[NUM_PQ_CENTROIDS * dim]` — for centroid `c` and
///   dimension `d` the value lives at `pq_table[c * dim + d]`.
/// - `chunk_offsets`: length `num_pq_chunks + 1`, marking the start/end
///   dimension of each chunk.
/// - `centroids`: per-dimension mean (length `dim`), subtracted during
///   preprocessing.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FixedChunkPQTable {
    /// Flat centroid table of size `[NUM_PQ_CENTROIDS * dim]`.
    pub pq_table: Vec<f32>,

    /// Variable chunk boundaries (length = num_pq_chunks + 1).
    pub chunk_offsets: Vec<usize>,

    /// Per-dimension centroid (mean), length = dim.
    pub centroids: Vec<f32>,

    /// Number of PQ chunks.
    num_pq_chunks: usize,

    /// Original vector dimensionality.
    dim: usize,
}

impl FixedChunkPQTable {
    // ------------------------------------------------------------------
    // Construction from pre-computed data (mirrors reference `new`)
    // ------------------------------------------------------------------

    /// Create a `FixedChunkPQTable` from pre-computed components.
    pub fn new(
        dim: usize,
        num_pq_chunks: usize,
        pq_table: Vec<f32>,
        centroids: Vec<f32>,
        chunk_offsets: Vec<usize>,
    ) -> Self {
        Self {
            pq_table,
            chunk_offsets,
            centroids,
            num_pq_chunks,
            dim,
        }
    }

    // ------------------------------------------------------------------
    // Training
    // ------------------------------------------------------------------

    /// Train a PQ table from raw data.
    ///
    /// `data` is a flat slice of `num_points * dim` floats stored row-major.
    pub fn train(data: &[f32], num_points: usize, dim: usize, num_pq_chunks: usize) -> Self {
        assert_eq!(
            data.len(),
            num_points * dim,
            "data length must equal num_points * dim"
        );
        assert!(num_pq_chunks > 0 && num_pq_chunks <= dim);

        // (a) Compute chunk_offsets — divide dim into roughly equal chunks.
        let chunk_offsets = compute_chunk_offsets(dim, num_pq_chunks);

        // (b) Compute per-dimension centroids (mean).
        let centroids = compute_centroids(data, num_points, dim);

        // (c) Subtract centroids from data (preprocessing).
        let mut centered: Vec<f32> = data.to_vec();
        for p in 0..num_points {
            let row = &mut centered[p * dim..(p + 1) * dim];
            for (val, &c) in row.iter_mut().zip(centroids.iter()) {
                *val -= c;
            }
        }

        // (d) + (e) For each chunk run k-means then populate pq_table.
        let mut pq_table = vec![0.0f32; NUM_PQ_CENTROIDS * dim];

        // Run k-means per chunk in parallel, collect results, then write into
        // pq_table sequentially (avoids mutable aliasing).
        let chunk_centroids: Vec<Vec<f32>> = (0..num_pq_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start_dim = chunk_offsets[chunk_idx];
                let end_dim = chunk_offsets[chunk_idx + 1];
                let sub_dim = end_dim - start_dim;

                // Extract sub-vectors for this chunk.
                let mut sub_data = vec![0.0f32; num_points * sub_dim];
                for p in 0..num_points {
                    let src = &centered[p * dim + start_dim..p * dim + end_dim];
                    let dst = &mut sub_data[p * sub_dim..(p + 1) * sub_dim];
                    dst.copy_from_slice(src);
                }

                run_kmeans_flat(&sub_data, num_points, sub_dim, NUM_PQ_CENTROIDS, 20)
            })
            .collect();

        // Populate pq_table.
        for chunk_idx in 0..num_pq_chunks {
            let start_dim = chunk_offsets[chunk_idx];
            let end_dim = chunk_offsets[chunk_idx + 1];
            let sub_dim = end_dim - start_dim;
            let ref_centroids = &chunk_centroids[chunk_idx];

            for centroid_idx in 0..NUM_PQ_CENTROIDS {
                for d in 0..sub_dim {
                    pq_table[centroid_idx * dim + start_dim + d] =
                        ref_centroids[centroid_idx * sub_dim + d];
                }
            }
        }

        Self {
            pq_table,
            chunk_offsets,
            centroids,
            num_pq_chunks,
            dim,
        }
    }

    // ------------------------------------------------------------------
    // Encoding
    // ------------------------------------------------------------------

    /// Encode data points into PQ codes.
    ///
    /// Returns a flat `Vec<u8>` of size `num_points * num_pq_chunks`.
    /// The data is centered (centroids subtracted) before encoding.
    pub fn encode(&self, data: &[f32], num_points: usize) -> Vec<u8> {
        assert_eq!(data.len(), num_points * self.dim);

        // Center the data first.
        let mut centered: Vec<f32> = data.to_vec();
        for p in 0..num_points {
            let row = &mut centered[p * self.dim..(p + 1) * self.dim];
            for (val, &c) in row.iter_mut().zip(self.centroids.iter()) {
                *val -= c;
            }
        }

        let mut codes = vec![0u8; num_points * self.num_pq_chunks];

        codes
            .par_chunks_mut(self.num_pq_chunks)
            .enumerate()
            .for_each(|(p, code_row)| {
                let point = &centered[p * self.dim..(p + 1) * self.dim];
                for chunk_idx in 0..self.num_pq_chunks {
                    let start_dim = self.chunk_offsets[chunk_idx];
                    let end_dim = self.chunk_offsets[chunk_idx + 1];
                    let sub_vec = &point[start_dim..end_dim];

                    let mut best_idx = 0u8;
                    let mut best_dist = f32::MAX;

                    for c in 0..NUM_PQ_CENTROIDS {
                        let mut dist = 0.0f32;
                        for d in start_dim..end_dim {
                            let diff = self.pq_table[c * self.dim + d] - sub_vec[d - start_dim];
                            dist += diff * diff;
                        }
                        if dist < best_dist {
                            best_dist = dist;
                            best_idx = c as u8;
                        }
                    }
                    code_row[chunk_idx] = best_idx;
                }
            });

        codes
    }

    // ------------------------------------------------------------------
    // Distance computation
    // ------------------------------------------------------------------

    /// Subtract centroids from `query_vec` in-place.
    pub fn preprocess_query(&self, query_vec: &mut [f32]) {
        for (q, &c) in query_vec.iter_mut().zip(self.centroids.iter()) {
            *q -= c;
        }
    }

    /// Build an ADC distance table of shape `[num_pq_chunks * NUM_PQ_CENTROIDS]`.
    ///
    /// Entry `[chunk * 256 + centroid]` is the squared L2 distance between
    /// `query_vec`'s sub-vector for that chunk and the given centroid.
    #[allow(clippy::needless_range_loop)]
    pub fn populate_chunk_distances(&self, query_vec: &[f32]) -> Vec<f32> {
        let mut dist_vec = vec![0.0f32; self.num_pq_chunks * NUM_PQ_CENTROIDS];
        for centroid_index in 0..NUM_PQ_CENTROIDS {
            for chunk_index in 0..self.num_pq_chunks {
                for dim_offset in
                    self.chunk_offsets[chunk_index]..self.chunk_offsets[chunk_index + 1]
                {
                    let diff = self.pq_table[self.dim * centroid_index + dim_offset]
                        - query_vec[dim_offset];
                    dist_vec[chunk_index * NUM_PQ_CENTROIDS + centroid_index] += diff * diff;
                }
            }
        }
        dist_vec
    }

    /// Compute L2 distance between a preprocessed query and a PQ-encoded
    /// base vector directly (without a pre-computed distance table).
    #[allow(clippy::needless_range_loop)]
    pub fn l2_distance(&self, query_vec: &[f32], base_vec: &[u8]) -> f32 {
        let mut total = 0.0f32;
        for chunk_index in 0..self.num_pq_chunks {
            let centroid_idx = base_vec[chunk_index] as usize;
            for dim_offset in self.chunk_offsets[chunk_index]..self.chunk_offsets[chunk_index + 1] {
                let diff =
                    self.pq_table[self.dim * centroid_idx + dim_offset] - query_vec[dim_offset];
                total += diff * diff;
            }
        }
        total
    }

    /// Compute ADC distance for a single encoded point using a pre-computed
    /// chunk-distance table.
    pub fn adc_distance(&self, codes: &[u8], pq_dists: &[f32]) -> f32 {
        let mut dist = 0.0f32;
        for chunk in 0..self.num_pq_chunks {
            let centroid_id = codes[chunk] as usize;
            dist += pq_dists[chunk * NUM_PQ_CENTROIDS + centroid_id];
        }
        dist
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Number of PQ chunks.
    pub fn get_num_chunks(&self) -> usize {
        self.num_pq_chunks
    }

    /// Vector dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    // ------------------------------------------------------------------
    // Serialization
    // ------------------------------------------------------------------

    /// Save the model to a file using bincode.
    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> ANNResult<()> {
        let mut file = BufWriter::new(File::create(path)?);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        bincode::serde::encode_into_std_write(self, &mut file, config)
            .map_err(|err| ANNError::log_serialize_error(err))?;
        Ok(())
    }

    /// Load a model from a file using bincode.
    pub fn load_model<P: AsRef<Path>>(path: P) -> ANNResult<Self> {
        let mut file = BufReader::new(File::open(path)?);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        let table: Self = bincode::serde::decode_from_std_read(&mut file, config)
            .map_err(|err| ANNError::log_deserialize_error(err))?;
        Ok(table)
    }
}

// ======================================================================
// Free functions
// ======================================================================

/// Batch PQ distance lookup.
///
/// Given `pq_ids` (flat `[n_pts * pq_nchunks]`) and a pre-computed distance
/// table `pq_dists` (flat `[pq_nchunks * 256]`), return a `Vec<f32>` of
/// length `n_pts` with the summed distances.
///
/// This is the safe-Rust equivalent of the reference implementation (no
/// `_mm_prefetch` / `unsafe`).
pub fn pq_dist_lookup(
    pq_ids: &[u8],
    n_pts: usize,
    pq_nchunks: usize,
    pq_dists: &[f32],
) -> Vec<f32> {
    let mut dists_out = vec![0.0f32; n_pts];
    for chunk in 0..pq_nchunks {
        let chunk_dists = &pq_dists[NUM_PQ_CENTROIDS * chunk..];
        for (n_iter, dist) in dists_out.iter_mut().enumerate() {
            let pq_centerid = pq_ids[pq_nchunks * n_iter + chunk] as usize;
            *dist += chunk_dists[pq_centerid];
        }
    }
    dists_out
}

// ======================================================================
// Internal helpers
// ======================================================================

/// Divide `dim` into `num_chunks` roughly-equal pieces.
/// Returns a vector of length `num_chunks + 1`.
fn compute_chunk_offsets(dim: usize, num_chunks: usize) -> Vec<usize> {
    let base = dim / num_chunks;
    let remainder = dim % num_chunks;
    let mut offsets = Vec::with_capacity(num_chunks + 1);
    offsets.push(0);
    for i in 0..num_chunks {
        let prev = *offsets.last().unwrap();
        // Distribute the remainder among the first `remainder` chunks.
        let chunk_size = if i < remainder { base + 1 } else { base };
        offsets.push(prev + chunk_size);
    }
    offsets
}

/// Compute per-dimension mean across all points.
fn compute_centroids(data: &[f32], num_points: usize, dim: usize) -> Vec<f32> {
    let mut centroids = vec![0.0f32; dim];
    for p in 0..num_points {
        let row = &data[p * dim..(p + 1) * dim];
        for (c, &v) in centroids.iter_mut().zip(row.iter()) {
            *c += v;
        }
    }
    let n = num_points as f32;
    for c in centroids.iter_mut() {
        *c /= n;
    }
    centroids
}

/// Run k-means on flat sub-vector data.
///
/// `data` is `[num_points * sub_dim]`, returns flat centroids `[k * sub_dim]`.
fn run_kmeans_flat(
    data: &[f32],
    num_points: usize,
    sub_dim: usize,
    k: usize,
    max_iters: usize,
) -> Vec<f32> {
    let mut rng = rand::rng();

    // Initialise centroids from random data points.
    let mut indices: Vec<usize> = (0..num_points).collect();
    indices.shuffle(&mut rng);

    let mut centroids = vec![0.0f32; k * sub_dim];
    for (i, &idx) in indices.iter().enumerate().take(k) {
        let src = &data[idx * sub_dim..(idx + 1) * sub_dim];
        centroids[i * sub_dim..(i + 1) * sub_dim].copy_from_slice(src);
    }

    // If fewer points than k, just return what we have (padded with zeros).
    if num_points < k {
        return centroids;
    }

    for _iter in 0..max_iters {
        // Assign each point to the nearest centroid.
        let assignments: Vec<usize> = (0..num_points)
            .into_par_iter()
            .map(|p| {
                let point = &data[p * sub_dim..(p + 1) * sub_dim];
                let mut best_idx = 0;
                let mut best_dist = f32::MAX;
                for c in 0..k {
                    let mut dist = 0.0f32;
                    let centroid = &centroids[c * sub_dim..(c + 1) * sub_dim];
                    for d in 0..sub_dim {
                        let diff = point[d] - centroid[d];
                        dist += diff * diff;
                    }
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = c;
                    }
                }
                best_idx
            })
            .collect();

        // Recompute centroids.
        let mut new_centroids = vec![0.0f32; k * sub_dim];
        let mut counts = vec![0usize; k];
        for (p, &cluster) in assignments.iter().enumerate() {
            let row = &data[p * sub_dim..(p + 1) * sub_dim];
            let dst = &mut new_centroids[cluster * sub_dim..(cluster + 1) * sub_dim];
            for d in 0..sub_dim {
                dst[d] += row[d];
            }
            counts[cluster] += 1;
        }

        for c in 0..k {
            if counts[c] > 0 {
                let row = &mut new_centroids[c * sub_dim..(c + 1) * sub_dim];
                let cnt = counts[c] as f32;
                for val in row.iter_mut() {
                    *val /= cnt;
                }
            } else {
                // Keep old centroid for empty clusters.
                let src_start = c * sub_dim;
                new_centroids[src_start..src_start + sub_dim]
                    .copy_from_slice(&centroids[src_start..src_start + sub_dim]);
            }
        }

        // Check for convergence.
        if new_centroids == centroids {
            break;
        }
        centroids = new_centroids;
    }

    centroids
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_and_encode() {
        let dim = 16;
        let n = 500;
        let data: Vec<f32> = (0..n * dim)
            .map(|i| (i as f32) / (n * dim) as f32)
            .collect();
        let pq = FixedChunkPQTable::train(&data, n, dim, 4);
        assert_eq!(pq.get_num_chunks(), 4);
        assert_eq!(pq.dim(), dim);
        assert_eq!(pq.pq_table.len(), NUM_PQ_CENTROIDS * dim);
        assert_eq!(pq.chunk_offsets.len(), 5); // 4 chunks + 1

        let codes = pq.encode(&data, n);
        assert_eq!(codes.len(), n * 4);
    }

    #[test]
    fn test_adc_distance_self_small() {
        let dim = 16;
        let n = 500;
        let data: Vec<f32> = (0..n * dim)
            .map(|i| (i as f32) / (n * dim) as f32)
            .collect();
        let pq = FixedChunkPQTable::train(&data, n, dim, 4);
        let codes = pq.encode(&data, n);

        let mut query = data[0..dim].to_vec();
        pq.preprocess_query(&mut query);
        let dists = pq.populate_chunk_distances(&query);
        let self_dist = pq.adc_distance(&codes[0..4], &dists);
        assert!(
            self_dist < 1.0,
            "Self distance {} should be small",
            self_dist
        );
    }

    #[test]
    fn test_pq_dist_lookup_batch() {
        let pq_ids: Vec<u8> = vec![1, 3, 2, 2];
        let mut pq_dists: Vec<f32> = Vec::with_capacity(256 * 2);
        for i in 0..512 {
            pq_dists.push(i as f32 * 0.01);
        }

        let dists_out = pq_dist_lookup(&pq_ids, 2, 2, &pq_dists);
        assert_eq!(dists_out.len(), 2);
        assert_eq!(dists_out[0], pq_dists[1] + pq_dists[256 + 3]);
        assert_eq!(dists_out[1], pq_dists[2] + pq_dists[256 + 2]);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dim = 16;
        let n = 300;
        let data: Vec<f32> = (0..n * dim).map(|i| i as f32).collect();
        let pq = FixedChunkPQTable::train(&data, n, dim, 4);

        let dir = std::env::temp_dir();
        let path = dir.join("test_pq_table_roundtrip.bin");
        pq.save_model(&path).unwrap();
        let loaded = FixedChunkPQTable::load_model(&path).unwrap();

        assert_eq!(pq.dim(), loaded.dim());
        assert_eq!(pq.get_num_chunks(), loaded.get_num_chunks());
        assert_eq!(pq.pq_table.len(), loaded.pq_table.len());
        let _ = std::fs::remove_file(&path);
    }
}
