/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Standalone binary for building CompressedDiskANN and generating clustering visualizations.
//!
//! Usage:
//!   cargo run -p staged-diskann --release --bin visualize -- \
//!     --base data/sift/sift_base.fvecs \
//!     --max-points 100000 \
//!     --output-dir visualizations/staged_diskann \
//!     --max-cluster-size 10 \
//!     --max-conn-clusters 4 \
//!     --max-conn-per-cluster 3 \
//!     --critical-min-rate 0.3

use diskann::model::vertex::{DIM_32, DIM_256};
use ndarray::Array2;
use rand::{RngExt, SeedableRng};
use staged_diskann::visualization::VISUALIZATION_DIMENSION;
use staged_diskann::{
    ClusteringMethod, DIM_128, DIM_960, DiskANN, StagedDiskANN, build_diskann_index,
};
use std::io::{self, Read as _};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn read_fvecs<P: AsRef<Path>>(path: P) -> io::Result<Vec<Vec<f32>>> {
    let mut file = io::BufReader::new(std::fs::File::open(path)?);
    let mut vectors = Vec::new();
    let mut dim_buf = [0u8; 4];

    loop {
        match file.read_exact(&mut dim_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;
        let mut data = vec![0u8; dim * 4];
        file.read_exact(&mut data)?;
        let vec: Vec<f32> = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        vectors.push(vec);
    }
    Ok(vectors)
}

// Generate a random dataset with brute-force ground truth.
///
/// - `num_points`: number of base vectors
/// - `dimension`: vector dimensionality
/// - `seed`: RNG seed for reproducibility
pub fn generate_random_dataset(
    num_points: usize,
    dimension: usize,
    seed: u64,
) -> anyhow::Result<Vec<Vec<f32>>> {
    log::info!(
        "Generating random dataset: {} points, dim={}",
        num_points,
        dimension,
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Generate base vectors
    Ok((0..num_points)
        .map(|_| {
            (0..dimension)
                .map(|_| rng.random_range(0f32..10f32))
                .collect()
        })
        .collect())
}

struct Config {
    base_path: String,
    max_points: usize,
    output_dir: String,
    random_dataset: bool,
    // DiskANN params
    alpha: f32,
    graph_degree: usize,
    search_list_size: usize,
    // Compressed params
    max_cluster_point_size: usize,
    critical_minimum_rate: f32,
    // Clustering
    clustering_method: ClusteringMethod,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            base_path: "data/sift/sift_base.fvecs".to_string(),
            max_points: 100_000,
            output_dir: "visualizations/staged_diskann".to_string(),
            random_dataset: false,
            alpha: 1.2,
            graph_degree: 32,
            search_list_size: 64,
            max_cluster_point_size: 16,
            critical_minimum_rate: 0.7,
            clustering_method: ClusteringMethod::Cohesive,
        }
    }
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--base" => {
                i += 1;
                config.base_path = args[i].clone();
            }
            "--max-points" => {
                i += 1;
                config.max_points = args[i].parse().expect("invalid --max-points");
            }
            "--output-dir" => {
                i += 1;
                config.output_dir = args[i].clone();
            }
            "--random-dataset" => {
                config.random_dataset = true;
            }
            "--alpha" => {
                i += 1;
                config.alpha = args[i].parse().expect("invalid --alpha");
            }
            "--graph-degree" => {
                i += 1;
                config.graph_degree = args[i].parse().expect("invalid --graph-degree");
            }
            "--search-list-size" => {
                i += 1;
                config.search_list_size = args[i].parse().expect("invalid --search-list-size");
            }
            "--max-cluster-size" => {
                i += 1;
                config.max_cluster_point_size =
                    args[i].parse().expect("invalid --max-cluster-size");
            }
            "--critical-min-rate" => {
                i += 1;
                config.critical_minimum_rate =
                    args[i].parse().expect("invalid --critical-min-rate");
            }
            "--clustering" => {
                i += 1;
                config.clustering_method = match args[i].as_str() {
                    "cohesive" => ClusteringMethod::Cohesive,
                    "lpa" => ClusteringMethod::LabelPropagation,
                    other => panic!("Unknown clustering method: {other}. Use 'cohesive' or 'lpa'."),
                };
            }
            "--help" | "-h" => {
                println!(
                    "CompressedDiskANN Visualization Tool\n\
                     \n\
                     Usage: visualize [OPTIONS]\n\
                     \n\
                     Options:\n\
                     --base <path>                  Base vectors file (.fvecs) [default: data/sift/sift_base.fvecs]\n\
                     --max-points <n>               Max points to load [default: 100000]\n\
                     --output-dir <dir>             Output directory [default: visualizations/staged_diskann]\n\
                     --random-dataset               Use random generated dataset [default: false]\n\
                     --alpha <f>                    DiskANN alpha [default: 1.2]\n\
                     --graph-degree <n>             Graph max degree [default: 32]\n\
                     --search-list-size <n>         Search list size L [default: 64]\n\
                     --max-cluster-size <n>         Max points per cluster [default: 10]\n\
                     --critical-min-rate <f>        Cluster merge threshold [default: 0.3]\n\
                     --clustering <method>          Clustering method: cohesive or lpa [default: cohesive]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }
    config
}

fn build_and_visualize_2(config: &Config, data_flat: Vec<f32>, num_points: usize) {
    let data_2d = Array2::from_shape_vec((num_points, VISUALIZATION_DIMENSION), data_flat)
        .expect("Failed to reshape data")
        .to_shared();

    println!("Building DiskANN (2-dim {} points)...", num_points);
    let start = Instant::now();
    let result = DiskANN::<2>::new(
        data_2d.clone(),
        2.0,
        32,
        32,
        None,
        false,
        None,
        None,
        None,
        false,
        true,
    );
    println!("DiskANN built in {:.2}s", start.elapsed().as_secs_f32(),);

    let candidate_sets = result.candidate_set_manager.candidate_sets;

    println!(
        "Building StagedDiskANN (cluster_size={}, crit_rate={}, method={:?})...",
        config.max_cluster_point_size, config.critical_minimum_rate, config.clustering_method
    );
    let start = Instant::now();
    let staged = StagedDiskANN::<2>::new(
        data_2d,
        result.graph,
        Arc::new(candidate_sets),
        result.entry_point,
        None,
        None,
        config.max_cluster_point_size,
        4,
        2,
        config.critical_minimum_rate,
        None,
        config.clustering_method,
        false,
    );
    println!(
        "StagedDiskANN built in {:.2}s",
        start.elapsed().as_secs_f32()
    );

    println!("Generating visualizations...");
    staged
        .generate_visualizations(&config.output_dir)
        .expect("Failed to generate visualizations");

    println!("Done! Visualizations saved to {}", config.output_dir);
}

fn build_and_visualize_32(config: &Config, data_flat: Vec<f32>, num_points: usize) {
    let data_2d = Array2::from_shape_vec((num_points, DIM_32), data_flat)
        .expect("Failed to reshape data")
        .to_shared();

    println!("Building DiskANN (32-dim {} points)...", num_points);
    let start = Instant::now();
    let result = build_diskann_index(
        &data_2d,
        config.alpha,
        config.graph_degree as u32,
        config.search_list_size as u32,
        false,
        None,
        None,
        true,
    );
    println!(
        "DiskANN built in {:.2}s (graph: {:.2}s, PQ: {:.2}s)",
        start.elapsed().as_secs_f32(),
        result.graph_build_time.as_secs_f32(),
        result.pq_build_time.as_secs_f32(),
    );

    println!(
        "Building CompressedDiskANN (cluster_size={}, crit_rate={}, method={:?})...",
        config.max_cluster_point_size, config.critical_minimum_rate, config.clustering_method
    );
    let start = Instant::now();
    let staged = StagedDiskANN::<DIM_32>::new(
        data_2d,
        result.graph,
        result.candidate_sets,
        result.entry_point,
        None,
        None,
        config.max_cluster_point_size,
        4,
        2,
        config.critical_minimum_rate,
        None,
        config.clustering_method,
        false,
    );
    println!(
        "CompressedDiskANN built in {:.2}s",
        start.elapsed().as_secs_f32()
    );

    println!("Generating visualizations...");
    staged
        .generate_visualizations(&config.output_dir)
        .expect("Failed to generate visualizations");

    println!("Done! Visualizations saved to {}", config.output_dir);
}

fn build_and_visualize_128(config: &Config, data_flat: Vec<f32>, num_points: usize) {
    let data_2d = Array2::from_shape_vec((num_points, DIM_128), data_flat)
        .expect("Failed to reshape data")
        .to_shared();

    println!("Building DiskANN (128-dim, {} points)...", num_points);
    let start = Instant::now();
    let result = build_diskann_index(
        &data_2d,
        config.alpha,
        config.graph_degree as u32,
        config.search_list_size as u32,
        false,
        None,
        None,
        true,
    );
    println!(
        "DiskANN built in {:.2}s (graph: {:.2}s, PQ: {:.2}s)",
        start.elapsed().as_secs_f32(),
        result.graph_build_time.as_secs_f32(),
        result.pq_build_time.as_secs_f32(),
    );

    println!(
        "Building CompressedDiskANN (cluster_size={}, crit_rate={}, method={:?})...",
        config.max_cluster_point_size, config.critical_minimum_rate, config.clustering_method
    );
    let start = Instant::now();
    let staged = StagedDiskANN::<DIM_128>::new(
        data_2d,
        result.graph,
        result.candidate_sets,
        result.entry_point,
        None,
        None,
        config.max_cluster_point_size,
        4,
        2,
        config.critical_minimum_rate,
        None,
        config.clustering_method,
        false,
    );
    println!(
        "CompressedDiskANN built in {:.2}s",
        start.elapsed().as_secs_f32()
    );

    println!("Generating visualizations...");
    staged
        .generate_visualizations(&config.output_dir)
        .expect("Failed to generate visualizations");

    println!("Done! Visualizations saved to {}", config.output_dir);
}

fn build_and_visualize_960(config: &Config, data_flat: Vec<f32>, num_points: usize) {
    let data_2d = Array2::from_shape_vec((num_points, DIM_256), data_flat)
        .expect("Failed to reshape data")
        .to_shared();

    println!("Building DiskANN (960-dim, {} points)...", num_points);
    let start = Instant::now();
    let result = build_diskann_index(
        &data_2d,
        config.alpha,
        config.graph_degree as u32,
        config.search_list_size as u32,
        true,
        None,
        None,
        true,
    );
    println!(
        "DiskANN built in {:.2}s (graph: {:.2}s, PQ: {:.2}s)",
        start.elapsed().as_secs_f32(),
        result.graph_build_time.as_secs_f32(),
        result.pq_build_time.as_secs_f32(),
    );

    println!(
        "Building CompressedDiskANN (cluster_size={}, crit_rate={}, method={:?})...",
        config.max_cluster_point_size, config.critical_minimum_rate, config.clustering_method
    );
    let start = Instant::now();
    let staged = StagedDiskANN::<DIM_960>::new(
        data_2d,
        result.graph,
        result.candidate_sets,
        result.entry_point,
        None,
        None,
        config.max_cluster_point_size,
        4,
        2,
        config.critical_minimum_rate,
        None,
        config.clustering_method,
        false,
    );
    println!(
        "CompressedDiskANN built in {:.2}s",
        start.elapsed().as_secs_f32()
    );

    println!("Generating visualizations...");
    staged
        .generate_visualizations(&config.output_dir)
        .expect("Failed to generate visualizations");

    println!("Done! Visualizations saved to {}", config.output_dir);
}

fn main() {
    env_logger::init();
    let config = parse_args();

    let vectors = if config.random_dataset {
        println!("Generating test data");
        // Dimension is 128 (the most commonly supported across all algorithms)
        let dimension = VISUALIZATION_DIMENSION;
        generate_random_dataset(config.max_points, dimension, 42)
            .expect("Failed to generate random data")
    } else {
        println!("Loading data from {}...", config.base_path);
        read_fvecs(&config.base_path).expect("Failed to read fvecs file")
    };
    let dimension = vectors.first().map(|v| v.len()).unwrap_or(0);
    let num_points = vectors.len().min(config.max_points);
    println!(
        "Loaded {} vectors, dimension={}, using first {} points",
        vectors.len(),
        dimension,
        num_points
    );

    let data_flat: Vec<f32> = vectors[..num_points]
        .iter()
        .flat_map(|v| v.iter().copied())
        .collect();

    match dimension {
        VISUALIZATION_DIMENSION => build_and_visualize_2(&config, data_flat, num_points),
        DIM_32 => build_and_visualize_32(&config, data_flat, num_points),
        DIM_128 => build_and_visualize_128(&config, data_flat, num_points),
        DIM_960 => build_and_visualize_960(&config, data_flat, num_points),
        _ => panic!(
            "Unsupported dimension: {}. Only 2, 32, 128 and 960 are supported.",
            dimension
        ),
    }
}
