/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Programmatic configuration. These defaults do not read YAML or environment overrides.
//! IndexOptions is the caller-facing aggregate. Its graph and cascade controls
//! are declared alongside it, before defaults and validated run resolution.
use crate::{
    config::cascade::{AdmissionChoice, Cascade, PrefilterChoice, RerankChoice, SearchMetric},
    config::plan::IndexPlan,
    config::ResolvedSweep,
    config::*,
    data::VectorFormat,
};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
/// Owned service configuration, validated by `Index::new/open`.
/// See [configuration examples and error bounds](crate::guide::configuration).
pub struct IndexOptions {
    /// Distance/normalization semantics, shared with file and query decoding.
    pub metric: SearchMetric,
    /// Resident element type; None follows the input representation.
    pub storage: Option<VectorStorageKind>,
    /// File format override; None infers it from the filename.
    pub base_format: Option<VectorFormat>,
    /// Graph source, capacities and construction recipe.
    pub graph: GraphConfig,
    /// Optional stage overrides; None fields use dimension/metric defaults.
    pub cascade: CascadeConfig,
    /// Writable build cache directory; concurrent builders need separate paths.
    pub cache_dir: PathBuf,
    /// Dedicated Rayon worker count, positive and within MAX_WORKERS.
    pub threads: usize,
    /// Convergence history length, in 1..=64.
    pub window_size: usize,
    /// Optional finite positive estimate limit in GiB; not a hard allocator cap.
    pub memory_budget_gib: Option<f64>,
    /// Optional nonzero prefix length; None uses every input row.
    pub max_points: Option<usize>,
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

/// Overrides use the same vocabulary and validation as the CLI cascade.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CascadeConfig {
    /// Prefilter identifier in the shared CLI vocabulary.
    pub prefilter: Option<String>,
    /// Admission identifier compatible with metric and resident storage.
    pub admission: Option<String>,
    /// Reranking identifier; native u8 uses none.
    pub rerank: Option<String>,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            metric: SearchMetric::L2,
            storage: None,
            base_format: None,
            graph: GraphConfig::default(),
            cascade: CascadeConfig::default(),
            cache_dir: "cache/orion-ann".into(),
            threads: orion_ann_core::BuildConfig::default().threads,
            window_size: 5,
            memory_budget_gib: None,
            max_points: None,
        }
    }
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

impl IndexOptions {
    pub(crate) fn resolve(
        &self,
        base: PathBuf,
        format: VectorFormat,
        dimension: usize,
        count: usize,
        content_id: Option<String>,
    ) -> Result<ResolvedRunConfig> {
        orion_ann_vector::PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(dimension)?;
        if count == 0 || count >= u32::MAX as usize || self.max_points == Some(0) {
            return Err("base must contain 1..u32::MAX points (excluding the sentinel)".into());
        }
        if self.threads > orion_ann_core::algorithm::search::MAX_WORKERS {
            return Err(
                Error::configuration("threads", "thread count exceeds MAX_WORKERS")
                    .with_detail("maximum", orion_ann_core::algorithm::search::MAX_WORKERS)
                    .with_detail("actual", self.threads),
            );
        }
        if self.threads == 0 || !(1..=64).contains(&self.window_size) {
            return Err("threads must be positive and window_size must be in 1..=64".into());
        }
        if self
            .memory_budget_gib
            .is_some_and(|v| !v.is_finite() || v <= 0.0)
        {
            return Err("memory_budget_gib must be finite and positive".into());
        }
        let graph = &self.graph;
        let storage = self.storage.unwrap_or(
            if matches!(format, VectorFormat::Bvecs | VectorFormat::U8bin) {
                VectorStorageKind::U8
            } else {
                VectorStorageKind::F32
            },
        );
        let mut policy = graph.source;
        if graph.staged_file.is_some() {
            if policy == GraphSourcePolicy::Rust {
                return Err("staged_file conflicts with graph source rust".into());
            }
            policy = GraphSourcePolicy::Parlayann;
        }
        let source = if policy == GraphSourcePolicy::Rust {
            GraphSource::Rust
        } else {
            GraphSource::Parlayann
        };
        let mut cascade = if storage == VectorStorageKind::U8 {
            Cascade {
                prefilter: PrefilterChoice::None,
                admission: AdmissionChoice::NativeL2U8,
                rerank: RerankChoice::None,
            }
        } else {
            Cascade::default_for_specified_dimension_and_metric(dimension, self.metric)
        };
        if let Some(v) = &self.cascade.prefilter {
            cascade.prefilter = v.parse()?;
        }
        if let Some(v) = &self.cascade.admission {
            cascade.admission = v.parse()?;
        }
        if let Some(v) = &self.cascade.rerank {
            cascade.rerank = v.parse()?;
        }
        cascade.validate_cascade_options(self.metric)?;
        let index_plan = IndexPlan::resolve(
            storage,
            dimension,
            self.metric,
            format,
            format,
            cascade,
            source,
        )?;
        let mut config = ResolvedRunConfig {
            dataset: "anonymous".into(),
            base,
            base_content_id: content_id,
            query: None,
            groundtruth: None,
            num_points: Some(count),
            num_queries: None,
            query_batch_size: 1024,
            replay_queries: false,
            base_format: format,
            query_format: format,
            index_plan,
            memory_budget_gib: self.memory_budget_gib,
            prepare_only: true,
            dimension,
            metric: self.metric,
            alpha: graph.alpha,
            graph_degree: graph.degree,
            build_l: graph.build_l,
            max_extra: graph.max_extra,
            window_size: self.window_size,
            graph_source: source,
            graph_source_policy: policy,
            graph_selection_reason: "explicit runtime configuration",
            staged_file: graph.staged_file.clone(),
            max_points: self.max_points.unwrap_or(count).min(count),
            sweep: ResolvedSweep {
                k: 1,
                #[cfg(any(feature = "benchmark", test))]
                profile: "runtime".into(),
                search_list_sizes: vec![64],
                threads: self.threads,
                #[cfg(any(feature = "benchmark", test))]
                trials: 1,
                calibration_samples: 200,
                calibration_l: 48,
            },
            cache_namespace: String::new(),
            cache_dir: self.cache_dir.clone(),
            staged_file_explicit: graph.staged_file.is_some(),
            local_pct: 60,
        };
        config.build_config().validate()?;
        config.cache_namespace = config.compute_cache_namespace();
        Ok(config)
    }
}
