/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Graph policy, caller recipe and translation to core construction options.
use super::{ResolvedRunConfig, SearchMetric};
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
/// Graph source and construction controls. See [configuration](crate::guide::configuration).
pub struct GraphConfig {
    /// Graph policy; auto prefers ParlayANN. Without the parlay feature only f32 can fall back to Rust.
    pub source: GraphSourcePolicy,
    /// Optional STAG v3 export; selects PA and conflicts with explicit Rust.
    pub staged_file: Option<PathBuf>,
    /// Maximum navigation degree, greater than zero.
    pub degree: u32,
    /// Construction candidate width, at least degree.
    pub build_l: usize,
    /// Finite Vamana pruning alpha, at least 1.0.
    pub alpha: f32,
    /// Maximum retained extra candidates per graph node.
    pub max_extra: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        let build = orion_ann_core::BuildConfig::default();
        Self {
            source: GraphSourcePolicy::Auto,
            staged_file: None,
            degree: build.degree,
            build_l: build.build_l,
            alpha: build.alpha,
            max_extra: build.max_extra,
        }
    }
}

impl GraphSourcePolicy {
    // Auto probes the PA candidate first; filesystem selection belongs to graph/.
    pub fn candidate(self) -> GraphSource {
        match self {
            Self::Rust => GraphSource::Rust,
            Self::Auto | Self::Parlayann => GraphSource::Parlayann,
        }
    }
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
}
