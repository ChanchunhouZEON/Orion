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
    /// Generic IndexOptions YAML; explicit CLI flags override individual fields.
    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Display name for reports/snapshots; defaults to anonymous (or the saved name).
    /// Does not select a dataset preset or change cache identity.
    #[arg(long)]
    pub dataset_name: Option<String>,

    /// Experimental supplier STAG v4 graph; preserves its entry and partitions.
    #[cfg(feature = "experimental-graph")]
    #[arg(long, conflicts_with_all = ["staged_file", "graph_source", "graph_degree", "build_l", "alpha", "max_extra"])]
    pub supplier_staged: Option<PathBuf>,

    /// Input base-vector file; required when building a fresh index.
    #[arg(long)]
    pub base: Option<PathBuf>,

    /// Input query file consumed for calibration and search.
    #[arg(long)]
    pub query: Option<PathBuf>,
    /// Maximum query vectors buffered from a stream per search batch.
    #[arg(long, default_value_t = 1024)]
    pub query_batch_size: usize,

    /// Optional exact-neighbor file used to report recall.
    #[arg(long)]
    pub groundtruth: Option<PathBuf>,
    /// Report search metrics without recall.
    #[arg(long, conflicts_with = "groundtruth")]
    pub no_groundtruth: bool,

    /// Encoding used to decode the base file.
    #[arg(long, value_enum)]
    pub base_format: Option<VectorFormat>,

    /// Encoding used to decode the query file.
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
    /// Override one object: base=mapped, graph=resident, l2-u8=auto, etc. Repeatable.
    #[arg(long = "object-storage", value_name = "OBJECT=MODE", value_parser = parse_object_storage)]
    pub object_storage: Vec<(orion_ann::StorageObject, orion_ann::StorageMode)>,
    /// Build/import and save graph cache, then exit before calibration/search.
    #[arg(long)]
    pub prepare_only: bool,
    /// Physical vector dimension. Inferred from the selected base format if omitted.
    #[arg(long)]
    pub dimension: Option<usize>,

    /// Original metric used by graph construction and scoring.
    #[arg(long, value_enum)]
    pub metric: Option<SearchMetric>,

    /// Prefilter stage choice; defaults are resolved by runtime.
    #[arg(long)]
    pub prefilter: Option<String>,
    /// Metric-specific admission: l2-u8/l2-kt, ip-i8/ip-i16, cosine-i8/cosine-i16, or pq-l2/pq-ip/pq-cosine.
    #[arg(long)]
    pub admission: Option<String>,
    /// Final stage: l2-f32, ip-f32, cosine-f32, l2-u16, or none.
    #[arg(long)]
    pub rerank: Option<String>,

    /// Robust-pruning relaxation factor for graph construction.
    #[arg(long)]
    pub alpha: Option<f32>,

    /// Maximum navigation degree of the constructed graph.
    #[arg(long)]
    pub graph_degree: Option<u32>,

    /// Construction beam capacity used when discovering neighbors.
    #[arg(long)]
    pub build_l: Option<usize>,

    /// Maximum retained non-navigation candidates per node.
    #[arg(long)]
    pub max_extra: Option<usize>,

    /// Window length used by adaptive convergence calibration.
    #[arg(long)]
    pub window_size: Option<usize>,
    /// Auto reuses PA or builds with ParlayANN; f32 can use Rust when ParlayANN is disabled.
    #[arg(long, value_enum)]
    pub graph_source: Option<GraphSourcePolicy>,

    /// Existing ParlayANN staged partitions to import instead of building.
    #[arg(long)]
    pub staged_file: Option<PathBuf>,
    /// Separate cache location for an alternate graph built from the same vectors.
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,

    /// Maximum base rows to load; also supplies preflight point count.
    #[arg(long)]
    pub max_points: Option<usize>,

    /// Number of neighbors requested per query.
    #[arg(long)]
    pub k: Option<usize>,

    /// Worker count used by this process-local index.
    #[arg(long)]
    pub threads: Option<usize>,

    /// Print the resolved configuration used for this run.
    #[arg(long)]
    pub print_config: bool,
}

/// Parses one OBJECT=MODE override through the shared runtime parsers and reports malformed CLI syntax.
fn parse_object_storage(
    value: &str,
) -> Result<(orion_ann::StorageObject, orion_ann::StorageMode), String> {
    let (object, mode) = value.split_once('=').ok_or("expected OBJECT=MODE")?;
    Ok((
        object
            .parse()
            .map_err(|e: orion_ann::Error| e.to_string())?,
        mode.parse().map_err(|e: orion_ann::Error| e.to_string())?,
    ))
}
