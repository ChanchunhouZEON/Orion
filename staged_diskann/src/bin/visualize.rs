/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Standalone binary for building StagedDiskANN and generating visualizations.
//!
//! Usage:
//!   cargo run -p staged-diskann --release --features visualization --bin visualize -- \
//!     --base data/sift/sift_base.fvecs \
//!     --max-points 1000 \
//!     --output-dir visualizations/staged_diskann

use diskann::index::InmemIndex;
use rand::{RngExt, SeedableRng};
use staged_diskann::visualization::VISUALIZATION_DIMENSION;
use staged_diskann::{DIM_128, DIM_960, StagedDiskANN, build_diskann_index};
use std::io::{self, Read as _};
use std::path::Path;
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

fn generate_random_dataset(num_points: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..num_points)
        .map(|_| {
            (0..dimension)
                .map(|_| rng.random_range(0f32..10f32))
                .collect()
        })
        .collect()
}

struct Config {
    base_path: String,
    max_points: usize,
    output_dir: String,
    random_dataset: bool,
    alpha: f32,
    graph_degree: usize,
    search_list_size: usize,
    base_local_count: usize,
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
            base_local_count: 16,
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
                config.max_points = args[i].parse().expect("invalid");
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
                config.alpha = args[i].parse().expect("invalid");
            }
            "--graph-degree" => {
                i += 1;
                config.graph_degree = args[i].parse().expect("invalid");
            }
            "--search-list-size" => {
                i += 1;
                config.search_list_size = args[i].parse().expect("invalid");
            }
            "--base-local-count" => {
                i += 1;
                config.base_local_count = args[i].parse().expect("invalid");
            }
            "--help" | "-h" => {
                println!("StagedDiskANN Visualization Tool\n\nUsage: visualize [OPTIONS]\n");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    config
}

/// Build StagedDiskANN and generate visualizations for a given dimension.
fn build_and_visualize<const N: usize>(config: &Config, flat_data: &[f32], num_points: usize)
where
    [f32; N]: vector::FullPrecisionDistance<f32, N>,
{
    println!("Building DiskANN ({N}-dim, {num_points} points)...");
    let start = Instant::now();
    let mut result = build_diskann_index(
        flat_data,
        num_points,
        N,
        config.alpha,
        config.graph_degree as u32,
        config.search_list_size as u32,
        false,
        None,
        None,
        true,
        config.base_local_count,
    )
    .expect("build failed");
    println!(
        "DiskANN built in {:.2}s (graph: {:.2}s)",
        start.elapsed().as_secs_f32(),
        result.graph_build_time.as_secs_f32(),
    );

    // Take dataset from InmemIndex for visualization.
    let dataset = {
        let idx = result
            .index
            .as_any_mut()
            .downcast_mut::<InmemIndex<f32, N>>()
            .expect("downcast failed");
        std::mem::replace(
            &mut idx.dataset,
            diskann::model::InmemDataset::new(0, 1.0).unwrap(),
        )
    };
    drop(result.index);

    println!(
        "Building StagedDiskANN (base_local={})...",
        config.base_local_count
    );
    let start = Instant::now();
    let staged = StagedDiskANN::<N>::new(
        dataset,
        result.graph,
        &result.candidate_sets,
        result.entry_point,
        config.base_local_count,
        None,
        None,
        None,
        false,
    );
    println!(
        "StagedDiskANN built in {:.2}s",
        start.elapsed().as_secs_f32()
    );

    println!("Done! Index built successfully.");
}

fn main() {
    env_logger::init();
    let config = parse_args();

    let vectors = if config.random_dataset {
        println!("Generating random dataset...");
        generate_random_dataset(config.max_points, VISUALIZATION_DIMENSION, 42)
    } else {
        println!("Loading data from {}...", config.base_path);
        read_fvecs(&config.base_path).expect("Failed to read fvecs file")
    };
    let dimension = vectors.first().map(|v| v.len()).unwrap_or(0);
    let num_points = vectors.len().min(config.max_points);
    println!(
        "Loaded {} vectors, dim={}, using {} points",
        vectors.len(),
        dimension,
        num_points
    );

    let flat_data: Vec<f32> = vectors[..num_points]
        .iter()
        .flat_map(|v| v.iter().copied())
        .collect();

    match dimension {
        VISUALIZATION_DIMENSION => {
            build_and_visualize::<VISUALIZATION_DIMENSION>(&config, &flat_data, num_points)
        }
        128 => build_and_visualize::<DIM_128>(&config, &flat_data, num_points),
        960 => build_and_visualize::<DIM_960>(&config, &flat_data, num_points),
        _ => panic!("Unsupported dimension: {dimension}. Supported: 2, 128, 960."),
    }
}
