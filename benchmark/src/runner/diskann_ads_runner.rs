/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! DiskANN + ADSampling runner.
//!
//! Identical graph/build pipeline to [`DiskANNRunner`], but:
//! 1. a fixed random orthogonal rotation is applied to the dataset before the
//!    index is built, and
//! 2. every search call rotates the query with the same matrix and uses the
//!    ADSampling (scaled partial-sum) early-abort distance kernel.
//!
//! Because L2 distance is rotation-invariant, the graph is structurally
//! identical to an unrotated build — only the distance-compute kernel differs
//! at search time, which isolates the ADSampling speedup.

use crate::report::table::BuildTiming;
use crate::runner::common::{AlgorithmRunner, SearchResult};
use adsampling::Rotator;
use diskann::index::{create_inmem_index, ANNInmemIndex};
use diskann::model::{IndexConfiguration, IndexWriteParametersBuilder};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use vector::Metric;

pub struct DiskANNAdsRunner {
    index: Option<Box<dyn ANNInmemIndex<f32>>>,
    dimension: usize,
    search_list_size: u32,
    graph_degree: u32,
    alpha: f32,
    ads_epsilon: f32,
    rotator: RotatorKind,
    temp_data_file: Option<PathBuf>,
}

/// Const-generic dispatch for the rotator. Each dataset dimension we support
/// gets its own variant; SIFT/GIST etc. pick the matching one at build time.
enum RotatorKind {
    None,
    Dim32(Box<Rotator<32>>),
    Dim100(Box<Rotator<100>>),
    Dim128(Box<Rotator<128>>),
    Dim960(Box<Rotator<960>>),
}

impl DiskANNAdsRunner {
    pub fn new(search_list_size: usize, graph_degree: u32, alpha: f32, ads_epsilon: f32) -> Self {
        Self {
            index: None,
            dimension: 0,
            search_list_size: search_list_size as u32,
            graph_degree,
            alpha,
            ads_epsilon,
            rotator: RotatorKind::None,
            temp_data_file: None,
        }
    }

    pub fn set_search_list_size(&mut self, sls: usize) {
        self.search_list_size = sls as u32;
    }

    fn write_temp_data_file(data: &[f32], num_points: usize, dimension: usize) -> PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("diskann_ads_bench_{}.bin", std::process::id()));
        let mut file = std::fs::File::create(&path).expect("temp file");
        file.write_all(&(num_points as i32).to_le_bytes()).unwrap();
        file.write_all(&(dimension as i32).to_le_bytes()).unwrap();
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        file.write_all(byte_slice).unwrap();
        file.flush().unwrap();
        path
    }

    fn rotate_query(&self, query: &[f32]) -> Vec<f32> {
        match &self.rotator {
            RotatorKind::None => query.to_vec(),
            RotatorKind::Dim32(r) => {
                let mut arr = [0.0f32; 32];
                arr.copy_from_slice(&query[..32]);
                r.apply(&arr).to_vec()
            }
            RotatorKind::Dim100(r) => {
                let mut arr = [0.0f32; 100];
                arr.copy_from_slice(&query[..100]);
                r.apply(&arr).to_vec()
            }
            RotatorKind::Dim128(r) => {
                let mut arr = [0.0f32; 128];
                arr.copy_from_slice(&query[..128]);
                r.apply(&arr).to_vec()
            }
            RotatorKind::Dim960(r) => {
                let mut arr = [0.0f32; 960];
                arr.copy_from_slice(&query[..960]);
                r.apply(&arr).to_vec()
            }
        }
    }
}

impl AlgorithmRunner for DiskANNAdsRunner {
    fn name(&self) -> &str {
        "DiskANN+ADS"
    }

    fn build(&mut self, data: &[f32], num_points: usize, dimension: usize) -> BuildTiming {
        self.dimension = dimension;
        let start = Instant::now();

        // Rotate the dataset once, up front.
        let mut rotated: Vec<f32> = data.to_vec();
        const ROTATION_SEED: u64 = 0xA05A_A05A;
        match dimension {
            32 => {
                let r = Box::new(Rotator::<32>::new(ROTATION_SEED));
                r.apply_batch_inplace(&mut rotated);
                self.rotator = RotatorKind::Dim32(r);
            }
            100 => {
                let r = Box::new(Rotator::<100>::new(ROTATION_SEED));
                r.apply_batch_inplace(&mut rotated);
                self.rotator = RotatorKind::Dim100(r);
            }
            128 => {
                let r = Box::new(Rotator::<128>::new(ROTATION_SEED));
                r.apply_batch_inplace(&mut rotated);
                self.rotator = RotatorKind::Dim128(r);
            }
            960 => {
                let r = Box::new(Rotator::<960>::new(ROTATION_SEED));
                r.apply_batch_inplace(&mut rotated);
                self.rotator = RotatorKind::Dim960(r);
            }
            d => panic!("Unsupported dimension for ADSampling: {d}"),
        }

        let temp_path = Self::write_temp_data_file(&rotated, num_points, dimension);
        self.temp_data_file = Some(temp_path.clone());

        let num_threads = rayon::current_num_threads() as u32;
        let write_params =
            IndexWriteParametersBuilder::new(self.search_list_size, self.graph_degree)
                .with_alpha(self.alpha)
                .with_num_threads(num_threads)
                .build();
        let config = IndexConfiguration::new(
            Metric::L2,
            dimension,
            dimension,
            num_points,
            false,
            0,
            false,
            0,
            1.0,
            write_params,
        );

        let mut index: Box<dyn ANNInmemIndex<f32>> =
            create_inmem_index(config).expect("Failed to create DiskANN index");
        index
            .build(temp_path.to_str().unwrap(), num_points)
            .expect("DiskANN build failed");

        self.index = Some(index);
        BuildTiming {
            graph_build: start.elapsed(),
            overhead: Duration::ZERO,
        }
    }

    fn search(&self, query: &[f32], k: usize) -> SearchResult {
        let index = self.index.as_ref().expect("Index not built");
        let rotated = self.rotate_query(query);
        let start = Instant::now();
        let mut indices = vec![0u32; k];
        index
            .search_adsampling(
                &rotated,
                k,
                self.search_list_size,
                self.ads_epsilon,
                &mut indices,
            )
            .expect("DiskANN+ADS search failed");
        let duration = start.elapsed();
        SearchResult {
            neighbors: indices,
            duration,
        }
    }

    fn memory_bytes(&self) -> usize {
        0
    }
}

impl Drop for DiskANNAdsRunner {
    fn drop(&mut self) {
        if let Some(ref path) = self.temp_data_file {
            let _ = std::fs::remove_file(path);
        }
    }
}
