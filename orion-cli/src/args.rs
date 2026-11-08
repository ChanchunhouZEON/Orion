/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Command-line syntax only. Resolution delegates common rules to IndexOptions.
use clap::Parser;
use orion_ann::execution::PreflightFormat;
use orion_ann::{GraphSourcePolicy, SearchMetric, VectorFormat, VectorStorageKind};
use std::path::PathBuf;

/// Production command-line inputs mapped onto the shared service configuration.
#[derive(Debug, Clone, Parser)]
#[command(name = "orion", about = "Shared Orion input and search configuration")]
pub struct Args {
    #[arg(long)]
    pub base: Option<PathBuf>,
    #[arg(long)]
    pub query: Option<PathBuf>,
    /// Maximum query vectors buffered from a stream per search batch.
    #[arg(long, default_value_t = 1024)]
    pub query_batch_size: usize,
    #[arg(long)]
    pub groundtruth: Option<PathBuf>,
    /// Report search metrics without recall.
    #[arg(long, conflicts_with = "groundtruth")]
    pub no_groundtruth: bool,
    #[arg(long, value_enum)]
    pub base_format: Option<VectorFormat>,
    #[arg(long, value_enum)]
    pub query_format: Option<VectorFormat>,
    /// Element representation used after loading the base vectors.
    ///
    /// This affects resident memory usage and may impose additional restrictions on
    /// the selected cascade and graph source.
    #[arg(long, value_enum)]
    pub vector_storage: Option<VectorStorageKind>,
    /// Estimate memory from file headers or explicit dimension/max-points; no payload loading.
    #[arg(long)]
    pub preflight: bool,
    /// Preflight presentation (auto: terminal table, redirected JSON).
    #[arg(long, value_enum, default_value = "auto", requires = "preflight")]
    pub preflight_format: PreflightFormat,
    /// Reject configurations whose known resident components exceed this budget.
    ///
    /// The value is expressed in GiB and is only a lower-bound check; temporary
    /// construction allocations are not guaranteed to fit within this limit.
    #[arg(long)]
    pub memory_budget_gib: Option<f64>,
    /// Disable automatic resource-bounded RAM locking.
    #[arg(long)]
    pub no_auto_pin: bool,
    /// Physical placement for base, graph and all sidecar payloads.
    #[arg(long = "storage-mode", value_enum)]
    pub storage_options: Option<orion_ann::StorageMode>,
    /// Existing disk-backed directory for private mapped payloads.
    #[arg(long)]
    pub mmap_dir: Option<PathBuf>,
    /// Build/import and save graph cache, then exit before calibration/search.
    #[arg(long)]
    pub prepare_only: bool,
    /// Physical vector dimension. Inferred from the selected base format if omitted.
    #[arg(long)]
    pub dimension: Option<usize>,
    #[arg(long, value_enum)]
    pub metric: Option<SearchMetric>,
    #[arg(long)]
    pub prefilter: Option<String>,
    #[arg(long)]
    pub admission: Option<String>,
    #[arg(long)]
    pub rerank: Option<String>,
    #[arg(long)]
    pub alpha: Option<f32>,
    #[arg(long)]
    pub graph_degree: Option<u32>,
    #[arg(long)]
    pub build_l: Option<usize>,
    #[arg(long)]
    pub max_extra: Option<usize>,
    #[arg(long)]
    pub window_size: Option<usize>,
    /// Auto reuses PA or builds with ParlayANN; f32 can use Rust when ParlayANN is disabled.
    #[arg(long, value_enum)]
    pub graph_source: Option<GraphSourcePolicy>,
    #[arg(long)]
    pub staged_file: Option<PathBuf>,
    /// Separate cache location for an alternate graph built from the same vectors.
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
    #[arg(long)]
    pub max_points: Option<usize>,
    #[arg(long)]
    pub k: Option<usize>,
    #[arg(long)]
    pub threads: Option<usize>,
    #[arg(long)]
    pub print_config: bool,
}
