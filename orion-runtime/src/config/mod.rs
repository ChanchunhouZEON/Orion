/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Validated configuration; independent of CLI flags, presets, and environment variables.
//! `options` defines caller inputs; `plan` validates representation/search
//! compatibility. The resolved run and sweep records below carry those decisions
//! to execution and driver reporting without parsing external inputs again.
// Preset/CLI resolution is optional; embedded callers use IndexOptions.

pub(crate) mod cascade;

pub mod options;
pub(crate) mod plan;

use crate::config::cascade::SearchMetric;
use crate::data::VectorFormat;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Source used to obtain the base proximity graph.
///
/// Construct with the ParlayANN or Rust builder, or import a STAG export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum GraphSource {
    /// Construct the graph using DiskANN native Rust implementation with extra
    /// candidates complemented.
    Rust,

    /// Import a ParlayANN graph or construct it with the in-memory backend.
    #[cfg_attr(feature = "cli", value(alias = "pa"))]
    Parlayann,
}

/// User policy, kept separate from the concrete source used in cache identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum GraphSourcePolicy {
    /// Prefer a PA cache/export, then build with ParlayANN. Without that feature, f32 can use Rust.
    Auto,
    Rust,
    #[cfg_attr(feature = "cli", value(alias = "pa"))]
    Parlayann,
}

/// Element representation used for resident base vectors.
///
/// This is independent of the on-disk [`VectorFormat`]. For example, vectors read
/// from a `.bvecs` file may either remain as native `u8` values or be expanded into
/// `f32` values after loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum VectorStorageKind {
    /// Store each coordinate as `f32`.
    F32,

    /// Keep each coordinate in its native unsigned 8-bit representation if the base vector
    /// is in `u8` stored form.
    U8,
}

#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[cfg(any(feature = "cli", feature = "benchmark"))]
pub enum PreflightFormat {
    /// Table for a terminal, JSON when stdout is redirected.
    #[default]
    Auto,
    Table,
    Json,
}

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
    #[serde(flatten)]
    pub(crate) index_plan: crate::config::plan::IndexPlan,
    pub memory_budget_gib: Option<f64>,
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
    /// Whether the staged graph path came explicitly from CLI or the legacy environment.
    ///
    /// This is excluded from serialization because it records resolution provenance
    /// rather than runtime semantics.
    #[serde(skip)]
    pub staged_file_explicit: bool,
    #[serde(skip)]
    pub local_pct: usize,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ResolvedSweep {
    pub k: usize,
    #[cfg(any(feature = "benchmark", test))]
    pub profile: String,
    pub search_list_sizes: Vec<usize>,
    pub threads: usize,
    #[cfg(any(feature = "benchmark", test))]
    pub trials: usize,
    pub calibration_samples: usize,
    pub calibration_l: usize,
}

impl ResolvedRunConfig {
    /// Translate service policy into the independent core construction recipe.
    /// File/cache selection stays here; the core builder has no sweep dependency.
    pub fn build_config(&self) -> orion_ann_core::BuildConfig {
        orion_ann_core::BuildConfig {
            backend: match self.graph_source {
                GraphSource::Rust => orion_ann_core::BuildBackend::DiskAnn,
                GraphSource::Parlayann => orion_ann_core::BuildBackend::ParlayAnn,
            },
            // Preserve existing cache/graph semantics: Rust builds L2, and
            // cosine rows have already been normalized by the input layer.
            metric: if self.graph_source == GraphSource::Parlayann
                && self.metric == SearchMetric::InnerProduct
            {
                orion_ann_core::BuildMetric::InnerProduct
            } else {
                orion_ann_core::BuildMetric::L2
            },
            degree: self.graph_degree,
            build_l: self.build_l,
            alpha: self.alpha,
            max_extra: self.max_extra,
            threads: self.sweep.threads,
            local_pct: if self.graph_source == GraphSource::Parlayann {
                self.local_pct
            } else {
                orion_ann_core::BuildConfig::default().local_pct
            },
        }
    }

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
