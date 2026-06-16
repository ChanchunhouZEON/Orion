/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::report::table::BuildTiming;
use crate::runner::common::{AlgorithmRunner, SearchResult};
use diskann::index::{create_inmem_index, ANNInmemIndex};
use diskann::model::{IndexConfiguration, IndexWriteParametersBuilder};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use vector::Metric;

pub struct DiskANNRunner {
    index: Option<Box<dyn ANNInmemIndex<f32>>>,
    dimension: usize,
    search_list_size: u32,
    graph_degree: u32,
    alpha: f32,
    /// Build-side distance metric. Defaults to `Metric::L2` for
    /// backward compatibility with the original constructor; set via
    /// [`Self::new_with_metric`] (or [`Self::set_metric`]) for
    /// Cosine builds on raw embeddings whose intrinsic metric is
    /// MIPS / cosine (msmarco_bert_1M, wiki_ada_1M, glove). The
    /// `diskann` core crate only ships L2 + Cosine, so MIPS workloads
    /// are approximated with Cosine on raw vectors (DiskANN normalises
    /// internally before its kernels).
    metric: Metric,
    /// Temp file path for data (diskann-core requires file-based loading)
    temp_data_file: Option<PathBuf>,
    /// Persistent cache file for Vamana graph + dataset. When set and file exists,
    /// `build()` skips Vamana construction and loads from disk instead.
    cache_path: Option<PathBuf>,
}

impl DiskANNRunner {
    pub fn set_search_list_size(&mut self, sls: usize) {
        self.search_list_size = sls as u32;
    }

    pub fn new(search_list_size: usize, graph_degree: u32, alpha: f32) -> Self {
        Self::new_with_metric(search_list_size, graph_degree, alpha, Metric::L2)
    }

    /// Construct with an explicit distance metric. The `diskann` core
    /// crate supports `L2` (squared Euclidean) and `Cosine`; for
    /// `Metric::Cosine` the kernel normalises input vectors before
    /// computing the dot product. Use this for MIPS-style datasets
    /// where the intended ranking is cosine-on-raw / dot-product.
    pub fn new_with_metric(
        search_list_size: usize,
        graph_degree: u32,
        alpha: f32,
        metric: Metric,
    ) -> Self {
        Self {
            index: None,
            dimension: 0,
            search_list_size: search_list_size as u32,
            graph_degree,
            alpha,
            metric,
            temp_data_file: None,
            cache_path: None,
        }
    }

    pub fn set_cache_path<P: Into<PathBuf>>(&mut self, path: P) {
        self.cache_path = Some(path.into());
    }

    /// Write flat f32 data to a temp file in diskann binary format:
    /// [i32: num_points] [i32: dimension] [f32 * num_points * dimension]
    fn write_temp_data_file(
        data: &[f32],
        num_points: usize,
        dimension: usize,
    ) -> std::io::Result<PathBuf> {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("diskann_bench_{}.bin", std::process::id()));
        let mut file = std::fs::File::create(&path)?;

        file.write_all(&(num_points as i32).to_le_bytes())?;
        file.write_all(&(dimension as i32).to_le_bytes())?;

        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        file.write_all(byte_slice)?;
        file.flush()?;

        Ok(path)
    }
}

impl AlgorithmRunner for DiskANNRunner {
    fn name(&self) -> &str {
        "DiskANN (Vamana)"
    }

    fn build(&mut self, data: &[f32], num_points: usize, dimension: usize) -> BuildTiming {
        self.dimension = dimension;
        let start = Instant::now();

        let num_threads = rayon::current_num_threads() as u32;
        let write_params =
            IndexWriteParametersBuilder::new(self.search_list_size, self.graph_degree)
                .with_alpha(self.alpha)
                .with_num_threads(num_threads)
                .build();

        let config = IndexConfiguration::new(
            self.metric,
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

        // Cache hit: load Vamana graph + data from disk, skip build entirely.
        if let Some(cp) = self.cache_path.as_ref() {
            let graph_file = cp.as_path();
            let data_file = cp.with_extension("bin.data");
            if graph_file.exists() && data_file.exists() {
                log::info!("Loading cached DiskANN index from {:?}", graph_file);
                index
                    .load(graph_file.to_str().unwrap(), num_points)
                    .expect("DiskANN cache load failed");
                self.index = Some(index);
                return BuildTiming {
                    graph_build: start.elapsed(),
                    overhead: Duration::ZERO,
                };
            }
        }

        // Cache miss: write temp data file, run Vamana build.
        let temp_path = Self::write_temp_data_file(data, num_points, dimension)
            .expect("Failed to write temp data file");
        self.temp_data_file = Some(temp_path.clone());

        index
            .build(temp_path.to_str().unwrap(), num_points)
            .expect("DiskANN build failed");

        if let Some(cp) = self.cache_path.as_ref() {
            if let Some(parent) = cp.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            match index.save(cp.to_str().unwrap()) {
                Ok(()) => log::info!("Saved DiskANN cache to {:?}", cp),
                Err(e) => log::warn!("Failed to save DiskANN cache: {}", e),
            }
        }

        let elapsed = start.elapsed();
        self.index = Some(index);
        BuildTiming {
            graph_build: elapsed,
            overhead: Duration::ZERO,
        }
    }

    fn search(&self, query: &[f32], k: usize) -> SearchResult {
        // Pure in-memory path. The mmap-search branch that used to
        // live here (gated on `self.mmap_state`) was dead in all
        // current benchmark modes — thread-sweep / qps-recall-sweep
        // never call `enable_mmap_search`, so the branch always took
        // the fall-through. Removing it keeps the hot path one
        // straight call into `ANNInmemIndex::search` and clarifies
        // that this runner is in-memory only. Re-introduce a separate
        // `MmapDiskANNRunner` if SSD search ever comes back.
        let index = self.index.as_ref().expect("Index not built");
        let start = Instant::now();
        let mut indices = vec![0u32; k];
        index
            .search(query, k, self.search_list_size, &mut indices)
            .expect("DiskANN search failed");
        let duration = start.elapsed();
        SearchResult {
            neighbors: indices,
            duration,
        }
    }

}

impl Drop for DiskANNRunner {
    fn drop(&mut self) {
        if let Some(ref path) = self.temp_data_file {
            let _ = std::fs::remove_file(path);
        }
    }
}
