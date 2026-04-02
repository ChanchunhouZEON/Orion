/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::report::table::BuildTiming;
use crate::runner::common::{AlgorithmRunner, SearchResult};
use nsg::index::{NSGIndex, NSGMmapIndex};
use nsg::model::{NSGConfig, Neighbor};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use vector::Metric;

pub struct NSGRunner {
    config: NSGConfig,
    metric: Metric,
    dimension: usize,
    inner: Option<NSGInner>,
}

enum NSGInner {
    Dim128(NSGIndex<f32, 128>),
    Dim960(NSGIndex<f32, 960>),
    Mmap128(NSGMmapIndex<128>),
    Mmap960(NSGMmapIndex<960>),
}

impl NSGRunner {
    pub fn new(r: usize, l: usize, c: usize, k: usize) -> Self {
        let config = NSGConfig {
            r,
            l,
            c,
            k,
            num_threads: 1,
        };
        Self {
            config,
            metric: Metric::L2,
            dimension: 0,
            inner: None,
        }
    }
}

impl AlgorithmRunner for NSGRunner {
    fn name(&self) -> &str {
        "NSG"
    }

    fn build(&mut self, data: &[f32], num_points: usize, dimension: usize) -> BuildTiming {
        self.dimension = dimension;
        let start = Instant::now();

        match dimension {
            128 => {
                let vectors = flat_to_arrays::<128>(data, num_points);
                let mut index =
                    NSGIndex::<f32, 128>::new(self.config.clone(), self.metric, num_points);
                index.build(vectors).expect("NSG build failed");
                self.inner = Some(NSGInner::Dim128(index));
            }
            960 => {
                let vectors = flat_to_arrays::<960>(data, num_points);
                let mut index =
                    NSGIndex::<f32, 960>::new(self.config.clone(), self.metric, num_points);
                index.build(vectors).expect("NSG build failed");
                self.inner = Some(NSGInner::Dim960(index));
            }
            _ => panic!("Unsupported dimension: {dimension}"),
        }

        BuildTiming { graph_build: start.elapsed(), overhead: Duration::ZERO }
    }

    fn search(&self, query: &[f32], k: usize) -> SearchResult {
        let start = Instant::now();
        let neighbors: Vec<Neighbor> = match self.inner.as_ref().expect("Index not built") {
            NSGInner::Dim128(index) => {
                let q = slice_to_array::<128>(query);
                index.search(&q, k).expect("NSG search failed")
            }
            NSGInner::Dim960(index) => {
                let q = slice_to_array::<960>(query);
                index.search(&q, k).expect("NSG search failed")
            }
            NSGInner::Mmap128(index) => {
                let q = slice_to_array::<128>(query);
                index.search(&q, k).expect("NSG mmap search failed")
            }
            NSGInner::Mmap960(index) => {
                let q = slice_to_array::<960>(query);
                index.search(&q, k).expect("NSG mmap search failed")
            }
        };
        let duration = start.elapsed();
        SearchResult {
            neighbors: neighbors.iter().map(|n| n.id as u32).collect(),
            duration,
        }
    }

    fn memory_bytes(&self) -> usize {
        0
    }

    fn supports_mmap(&self) -> bool {
        true
    }

    fn save_mmap(&self, dir: &Path) -> anyhow::Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("nsg_graph.anns");
        let path_str = path.to_str().unwrap();
        match self.inner.as_ref().expect("Index not built") {
            NSGInner::Dim128(index) => {
                NSGMmapIndex::save_mmap(index, path_str)?;
            }
            NSGInner::Dim960(index) => {
                NSGMmapIndex::save_mmap(index, path_str)?;
            }
            _ => anyhow::bail!("Already in mmap mode"),
        }
        Ok(path)
    }

    fn enable_mmap_search(&mut self, graph_path: &Path) -> anyhow::Result<()> {
        let path_str = graph_path.to_str().unwrap();
        match self.dimension {
            128 => {
                let mmap = NSGMmapIndex::<128>::load_mmap(path_str)?;
                self.inner = Some(NSGInner::Mmap128(mmap));
            }
            960 => {
                let mmap = NSGMmapIndex::<960>::load_mmap(path_str)?;
                self.inner = Some(NSGInner::Mmap960(mmap));
            }
            _ => anyhow::bail!("Unsupported dimension: {}", self.dimension),
        }
        Ok(())
    }

    fn warm_cache(&self, max_hops: usize) {
        match self.inner.as_ref() {
            Some(NSGInner::Mmap128(index)) => index.warm_cache(max_hops),
            Some(NSGInner::Mmap960(index)) => index.warm_cache(max_hops),
            _ => {}
        }
    }
}

fn flat_to_arrays<const N: usize>(data: &[f32], num_points: usize) -> Vec<[f32; N]> {
    assert_eq!(data.len(), num_points * N);
    let mut vectors = Vec::with_capacity(num_points);
    for i in 0..num_points {
        let mut arr = [0.0f32; N];
        arr.copy_from_slice(&data[i * N..(i + 1) * N]);
        vectors.push(arr);
    }
    vectors
}

fn slice_to_array<const N: usize>(slice: &[f32]) -> [f32; N] {
    assert!(slice.len() >= N);
    let mut arr = [0.0f32; N];
    arr.copy_from_slice(&slice[..N]);
    arr
}
