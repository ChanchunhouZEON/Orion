/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resolved execution input and stable graph-cache identity.
use super::{GraphSource, GraphSourcePolicy, ResolvedSweep, SearchMetric, VectorStorageKind};
use crate::data::VectorFormat;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize)]
pub struct ResolvedRunConfig {
    /// Preset name, or "anonymous" for explicit inputs without a preset.
    pub dataset: String,
    pub base: PathBuf,
    /// Identity of the authenticated base, when available after input preparation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_content_id: Option<String>,
    pub query: Option<PathBuf>,
    pub groundtruth: Option<PathBuf>,
    /// Preset source counts; discarded when the corresponding input is overridden.
    pub num_points: Option<usize>,
    pub num_queries: Option<usize>,
    pub query_batch_size: usize,
    pub replay_queries: bool,
    pub base_format: VectorFormat,
    pub query_format: VectorFormat,
    // Executable interpretation of the public, serialized run settings.
    // Constructed only after compatibility validation; consumers use the accessor.
    #[cfg(feature = "benchmark")]
    #[serde(flatten)]
    #[doc(hidden)]
    pub index_plan: crate::config::plan::IndexPlan,
    #[cfg(not(feature = "benchmark"))]
    #[serde(flatten)]
    pub(crate) index_plan: crate::config::plan::IndexPlan,
    pub memory_budget_gib: Option<f64>,
    pub auto_pin: bool,
    #[serde(skip_serializing_if = "crate::StorageOptions::is_default")]
    pub storage_options: crate::StorageOptions,
    pub prepare_only: bool,
    pub dimension: usize,
    pub metric: SearchMetric,
    pub alpha: f32,
    pub graph_degree: u32,
    pub build_l: usize,
    pub max_extra: usize,
    pub window_size: usize,
    pub graph_source: GraphSource,
    pub graph_source_policy: GraphSourcePolicy,
    /// Why automatic selection chose this source; explicit policies stay strict.
    pub graph_selection_reason: &'static str,
    pub staged_file: Option<PathBuf>,
    pub max_points: usize,
    pub sweep: ResolvedSweep,
    /// Cache namespace derived from graph-affecting inputs and construction settings.
    pub cache_namespace: String,
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

    pub(crate) fn base_representation(&self) -> crate::config::plan::BaseRepresentation {
        self.index_plan.base()
    }
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
            self.metric,
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
