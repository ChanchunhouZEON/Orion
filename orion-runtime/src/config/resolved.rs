/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resolved execution input and stable graph-cache identity.
use super::{GraphSource, GraphSourcePolicy, ResolvedSweep, SearchMetric, VectorStorageKind};
use crate::data::VectorFormat;
use serde::Serialize;
use std::path::PathBuf;

/// Validated runtime settings and graph identity after public input resolution.
#[derive(Debug, Clone, Serialize)]
pub struct ResolvedRunConfig {
    /// Execution-local allocation domain; excluded from identities and snapshots.
    #[cfg(feature = "benchmark")]
    #[doc(hidden)]
    #[serde(skip)]
    pub allocation_budget: std::sync::Arc<
        std::sync::OnceLock<orion_ann_platform::allocation_budget::AllocationBudget>,
    >,

    /// Execution-local allocation domain shared by actual buffer owners, excluded from persistence identity.
    #[cfg(not(feature = "benchmark"))]
    #[serde(skip)]
    pub(crate) allocation_budget: std::sync::Arc<
        std::sync::OnceLock<orion_ann_platform::allocation_budget::AllocationBudget>,
    >,
    /// Display name, or "anonymous" when none was supplied.
    pub dataset: String,

    /// Base input location used for loading and pre-content-identity lookup.
    pub base: PathBuf,
    /// Identity of the authenticated base, when available after input preparation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_content_id: Option<String>,

    /// Optional file query source for execution adapters.
    pub query: Option<PathBuf>,

    /// Optional evaluation input; search itself does not require ground truth.
    pub groundtruth: Option<PathBuf>,
    /// Preset source counts; discarded when the corresponding input is overridden.
    pub num_points: Option<usize>,

    /// Known query count when available for execution and memory forecasting.
    pub num_queries: Option<usize>,

    /// Maximum streamed query rows buffered by an execution adapter.
    pub query_batch_size: usize,

    /// Whether an adapter retains queries for repeated execution instead of one-pass streaming.
    pub replay_queries: bool,

    /// Resolved decoder for the original base file.
    pub base_format: VectorFormat,

    /// Resolved decoder for file-backed query batches.
    pub query_format: VectorFormat,
    // Executable interpretation of the public, serialized run settings.
    // Constructed only after compatibility validation; consumers use the accessor.
    /// Validated base and cascade interpretation consumed by preparation and search.
    #[cfg(feature = "benchmark")]
    #[serde(flatten)]
    #[doc(hidden)]
    pub index_plan: crate::config::plan::IndexPlan,

    /// Validated base and cascade interpretation consumed by preparation and search.
    #[cfg(not(feature = "benchmark"))]
    #[serde(flatten)]
    pub(crate) index_plan: crate::config::plan::IndexPlan,

    /// Optional managed-memory planning ceiling expressed in GiB, not a process RSS limit.
    pub memory_budget_gib: Option<f64>,

    /// Whether preparation should attempt bounded automatic page locking.
    pub auto_pin: bool,

    /// Deployment placement overrides; independent of graph content identity.
    #[serde(skip_serializing_if = "crate::StorageOptions::is_default")]
    pub storage_options: crate::StorageOptions,

    /// Whether this execution stops after graph preparation instead of preparing search sidecars.
    pub prepare_only: bool,

    /// Logical base and query width before padding.
    pub dimension: usize,

    /// Original search metric used to resolve encoding and score semantics.
    pub metric: SearchMetric,

    /// Robust-pruning parameter contributing to graph identity.
    pub alpha: f32,

    /// Maximum navigation degree requested from the builder.
    pub graph_degree: u32,

    /// Construction candidate beam capacity contributing to graph identity.
    pub build_l: usize,

    /// Maximum residual candidates retained per graph node.
    pub max_extra: usize,

    /// Convergence observation-window length for adaptive search.
    pub window_size: usize,

    /// Resolved graph provider after automatic or explicit source selection.
    pub graph_source: GraphSource,

    /// Original source-selection policy, preserving strict explicit requests.
    pub graph_source_policy: GraphSourcePolicy,
    /// Why automatic selection chose this source; explicit policies stay strict.
    pub graph_selection_reason: &'static str,

    /// Optional staged graph export selected for import.
    pub staged_file: Option<PathBuf>,

    /// Upper bound on loaded base rows.
    pub max_points: usize,

    /// Execution controls kept separate from graph construction identity.
    pub sweep: ResolvedSweep,
    /// Cache namespace derived from graph-affecting inputs and construction settings.
    pub cache_namespace: String,

    /// Working lookup directory; immutable payloads live in the associated object store.
    pub cache_dir: PathBuf,
    /// Explicit experiment policy; enabling the Cargo feature alone changes nothing.
    #[cfg(feature = "benchmark")]
    #[serde(skip)]
    pub benchmark_trust_cache: bool,
    /// Selected standalone metadata (.bin) or content-addressed record (.cache.json).
    #[cfg(feature = "benchmark")]
    #[serde(skip)]
    pub benchmark_cache_file: Option<PathBuf>,
    /// Whether the staged graph path came explicitly from CLI or the legacy environment.
    ///
    /// This is excluded from serialization because it records resolution provenance
    /// rather than runtime semantics.
    #[serde(skip)]
    pub staged_file_explicit: bool,

    /// Local-navigation partition proportion used by ParlayANN construction.
    #[serde(skip)]
    pub local_pct: usize,
}

impl ResolvedRunConfig {
    /// Resident coordinate type, independent of search stages and file format.
    pub fn storage(&self) -> VectorStorageKind {
        self.index_plan.base().kind()
    }

    /// Resolved candidate-evaluation recipe for driver and benchmark adapters.
    pub fn cascade(&self) -> crate::config::cascade::Cascade {
        self.index_plan.cascade()
    }

    /// Returns the validated base representation independently of cascade choices.
    pub(crate) fn base_representation(&self) -> crate::config::plan::BaseRepresentation {
        self.index_plan.base()
    }

    /// Returns the validated executable interpretation without re-parsing configuration.
    pub(crate) fn index_plan(&self) -> crate::config::plan::IndexPlan {
        self.index_plan
    }

    /// Computes the stable namespace used for graph-cache files.
    ///
    /// The identity includes the base-file identity and every resolved setting that
    /// changes the stored graph. Query paths and search-only sweep settings are
    /// intentionally excluded.
    ///
    pub fn compute_cache_namespace(&self) -> String {
        use sha2::{Digest, Sha256};
        let descriptor = serde_json::to_vec(&(
            "orion/graph-recipe",
            &self.base_content_id,
            // Resolution without input bytes is also used by offline preflight.
            self.base_content_id.is_none().then_some(&self.base),
            self.dimension,
            self.base_format,
            self.storage(),
            self.metric.identity_key(),
            self.alpha.to_bits(),
            self.graph_degree,
            self.build_l,
            self.max_extra,
            self.graph_source,
            (self.graph_source == GraphSource::Parlayann)
                .then_some((self.local_pct, crate::graph::PARLAY_RECIPE)),
        ))
        .expect("serialize graph recipe");
        format!("{:x}", Sha256::digest(descriptor))
    }
}
