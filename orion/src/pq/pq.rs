/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use diskann::common::ANNResult;
use ndarray::{ArcArray2, prelude::*};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct ProductQuantizer {
    dim: usize,
    m: usize,
    n_bits: u32,
    /// Kernel size
    ks: usize,
    sub_dim: usize,
    pub(super) codebooks: Vec<ArcArray2<f32>>,
    codes: ArcArray2<u8>,
}

// Prototype PQ implementation kept as reference; production search uses
// the codec-specific quantizers in `model::dataset::*` (L2U8, MipsI8,
// RaBitQ). Methods are exercised by the `mod tests` block at the bottom
// of this file but otherwise have no in-tree callers.
#[allow(dead_code)]
impl ProductQuantizer {
    pub fn new(n_samples: usize, dim: usize, n_subquantizers: usize, n_bits: u32) -> Self {
        assert_eq!(dim % n_subquantizers, 0, "Dimension must be divisible by M");
        let ks = (1 << n_bits) as usize;
        let sub_dim = dim / n_subquantizers;

        assert!(
            n_samples > ks,
            "sample size should be larger than kernel size"
        );

        Self {
            dim,
            m: n_subquantizers,
            n_bits,
            ks,
            sub_dim,
            codebooks: Vec::new(),
            codes: ArcArray2::<u8>::zeros((n_samples, n_subquantizers)),
        }
    }

    /// Simple KMeans Implementation
    fn run_kmeans(data: ArrayView2<f32>, k: usize, max_iters: usize) -> ArcArray2<f32> {
        let n_samples = data.nrows();
        let sub_dim = data.ncols();
        let mut rng = rand::rng();

        // Initialize random centroid points with size `k`
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);
        let mut centroids = ArcArray2::zeros((k, sub_dim));
        for (i, &index) in indices.iter().enumerate().take(k) {
            centroids.row_mut(i).assign(&data.row(index));
        }

        for _ in 0..max_iters {
            // Parallelism allocate the affiliation
            let assignments: Vec<usize> = data
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|point| {
                    centroids
                        .axis_iter(Axis(0))
                        .enumerate()
                        .map(|(idx, center)| {
                            let dist = point
                                .iter()
                                .zip(center.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f32>();
                            (idx, dist)
                        })
                        .min_by(|&a, b| a.1.partial_cmp(&b.1).unwrap())
                        .unwrap()
                        .0
                })
                .collect();

            // Update the centroid information
            let mut new_centroids = ArcArray2::<f32>::zeros((k, sub_dim));
            let mut counts = vec![0usize; k];
            for (i, &cluster_idx) in assignments.iter().enumerate() {
                new_centroids
                    .row_mut(cluster_idx)
                    .zip_mut_with(&data.row(i), |a, &b| *a += b);
                counts[cluster_idx] += 1;
            }

            for (i, &count) in counts.iter().enumerate().take(k) {
                if count > 0 {
                    new_centroids
                        .row_mut(i)
                        .map_inplace(|x| *x /= counts[i] as f32);
                }

                if centroids == new_centroids {
                    break;
                }
            }

            centroids = new_centroids;
        }

        centroids
    }

    pub fn train(&mut self, data: ArcArray2<f32>) {
        self.codebooks = (0..self.m)
            .into_par_iter()
            .map(|m| {
                let start = m * self.sub_dim;
                let end = (m + 1) * self.sub_dim;
                let sub_vectors = data.slice(s![.., start..end]);
                Self::run_kmeans(sub_vectors, self.ks, 20)
            })
            .collect();
    }

    pub fn encode(&mut self, data: ArcArray2<f32>) {
        ndarray::Zip::from(data.rows())
            .and(self.codes.rows_mut())
            .par_for_each(|vec, mut code_row| {
                for m in 0..self.m {
                    let start = m * self.sub_dim;
                    let end = (m + 1) * self.sub_dim;
                    let sub_vector = vec.slice(s![start..end]);
                    let centroid = &self.codebooks[m];

                    let (best_idx, _) = centroid
                        .axis_iter(Axis(0))
                        .enumerate()
                        .map(|(idx, center)| {
                            let d = sub_vector
                                .iter()
                                .zip(center.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f32>();
                            (idx, d)
                        })
                        .min_by(|&a, b| a.1.partial_cmp(&b.1).unwrap())
                        .unwrap();

                    code_row[m] = best_idx as u8;
                }
            });
    }

    pub fn compute_adc_table(&self, query: &ArrayView1<f32>) -> Vec<Array1<f32>> {
        (0..self.m)
            .into_par_iter()
            .map(|m| {
                let start = m * self.sub_dim;
                let end = (m + 1) * self.sub_dim;
                let qsub = query.slice(s![start..end]);
                let centroids = &self.codebooks[m];

                Array1::from_vec(
                    centroids
                        .axis_iter(Axis(0))
                        .map(|center| {
                            qsub.iter()
                                .zip(center.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum()
                        })
                        .collect(),
                )
            })
            .collect()
    }

    pub fn adc_distance(&self, code: usize, tables: &[Array1<f32>]) -> f32 {
        self.codes
            .row(code)
            .iter()
            .enumerate()
            .map(|(m, &c)| tables[m][c as usize])
            .sum()
    }

    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> ANNResult<()> {
        let mut file = BufWriter::new(File::create(path)?);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        bincode::serde::encode_into_std_write(self, &mut file, config)?;
        Ok(())
    }

    pub fn load_model<P: AsRef<Path>>(path: P) -> ANNResult<Self> {
        let mut file = BufReader::new(File::open(path)?);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        let pq = bincode::serde::decode_from_std_read(&mut file, config)?;
        Ok(pq)
    }
}

impl PartialEq for ProductQuantizer {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim && self.n_bits == other.n_bits && self.m == other.m
    }
}

impl Eq for ProductQuantizer {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};
    use std::fs;
    use std::path::PathBuf;

    const PQ_SAVE_DIR: &str = "tests/data";
    const PQ_SAVE_PATH: &str = "dskann_pq.bin";

    /// Helper function to create synthetic data for testing
    fn generate_mock_data(n: usize, dim: usize) -> ArcArray2<f32> {
        Array2::from_shape_fn((n, dim), |(i, j)| (i + j) as f32).into_shared()
    }

    #[test]
    fn test_new_initialization() {
        let dim = 128;
        let m = 8;
        let n_bits = 8;
        let n_samples = 300;

        let pq = ProductQuantizer::new(n_samples, dim, m, n_bits);

        // Check if dimensions and sub-dimensions are calculated correctly
        assert_eq!(pq.dim, dim);
        assert_eq!(pq.m, m);
        assert_eq!(pq.sub_dim, dim / m);
        assert_eq!(pq.ks, 256); // 1 << 8
        assert_eq!(pq.codes.dim(), (n_samples, m));
    }

    #[test]
    #[should_panic(expected = "Dimension must be divisible by M")]
    fn test_new_invalid_dimensions() {
        // 128 is not divisible by 9
        ProductQuantizer::new(100, 128, 9, 8);
    }

    #[test]
    fn test_train_and_encode() {
        let n_samples = 20;
        let dim = 4;
        let m = 2; // sub_dim = 2
        let n_bits = 2; // ks = 4

        let mut pq = ProductQuantizer::new(n_samples, dim, m, n_bits);
        let data = generate_mock_data(n_samples, dim);

        // Test Training: Codebooks should be populated
        pq.train(data.clone());
        assert_eq!(pq.codebooks.len(), m);
        for book in &pq.codebooks {
            assert_eq!(book.dim(), (4, 2)); // ks, sub_dim
        }

        // Test Encoding: Codes should be updated
        pq.encode(data.clone());
        for &code in pq.codes.iter() {
            // With 2 bits, codes must be in range [0, 3]
            assert!(code < 4);
        }
    }

    #[test]
    fn test_adc_computation() {
        let n_samples = 300;
        let dim = 4;
        let m = 2;
        let mut pq = ProductQuantizer::new(n_samples, dim, m, 8);
        let data = generate_mock_data(n_samples, dim);

        pq.train(data.clone());
        pq.encode(data.clone());

        let query = array![1.0, 2.0, 3.0, 4.0];

        // Compute distance table for the query
        let tables = pq.compute_adc_table(&query.view());
        assert_eq!(tables.len(), m);
        assert_eq!(tables[0].len(), 256);

        // Calculate distance to the first stored vector
        let dist = pq.adc_distance(0, &tables);

        // Distance should be non-negative
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_serialization_roundtrip() -> ANNResult<()> {
        let dim = 8;
        let m = 2;
        let n_samples = 300;
        let mut pq = ProductQuantizer::new(n_samples, dim, m, 8);
        let data = generate_mock_data(n_samples, dim);

        pq.train(data.clone());
        pq.encode(data.clone());

        // Create a temporary file path
        let dir = PathBuf::from(PQ_SAVE_DIR);
        fs::create_dir_all(&dir)?;
        let path = dir.join(PQ_SAVE_PATH);

        // Save the model
        pq.save_model(&path)?;

        // Load the model
        let loaded_pq = ProductQuantizer::load_model(&path)?;

        // Check if critical parameters match
        assert_eq!(pq.dim, loaded_pq.dim);
        assert_eq!(pq.m, loaded_pq.m);
        assert_eq!(pq.n_bits, loaded_pq.n_bits);

        // Check if codebooks are preserved
        assert_eq!(pq.codebooks.len(), loaded_pq.codebooks.len());
        for i in 0..m {
            assert_eq!(pq.codebooks[i], loaded_pq.codebooks[i]);
        }

        // Clear the temp file
        fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
