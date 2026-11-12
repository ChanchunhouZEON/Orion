/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Programmatic configuration. These defaults do not read YAML or environment overrides.
//! IndexOptions aggregates property-specific controls and resolves service input.
//! Graph and cascade definitions live beside their own defaults and rules.
use super::{
    cascade::{Cascade, SearchMetric},
    plan::IndexPlan,
    GraphSourcePolicy, ResolvedRunConfig, ResolvedSweep, VectorStorageKind,
};
use crate::{data::VectorFormat, Result};

// Preserve existing public paths while each property owns its implementation.
pub use super::cascade::CascadeConfig;
pub use super::graph::GraphConfig;
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
    /// Apply bounded, best-effort RAM locking during search preparation.
    /// Calibration and calibrated snapshot open perform this automatically;
    /// ordinary searches reuse the retained locks. No CLI feature is required.
    pub auto_pin: bool,
    /// Placement selected before base, graph and sidecar payload allocation.
    #[serde(skip_serializing_if = "crate::StorageOptions::is_default")]
    pub storage_options: crate::StorageOptions,
    /// Optional nonzero prefix length; None uses every input row.
    pub max_points: Option<usize>,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            metric: SearchMetric::SquaredL2,
            storage: None,
            base_format: None,
            graph: GraphConfig::default(),
            cascade: CascadeConfig::default(),
            cache_dir: "cache/orion-ann".into(),
            threads: orion_ann_core::BuildConfig::default().threads,
            window_size: 5,
            memory_budget_gib: None,
            auto_pin: true,
            storage_options: Default::default(),
            max_points: None,
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
        if count == 0 || count >= u32::MAX as usize || self.max_points == Some(0) {
            return Err("base must contain 1..u32::MAX points (excluding the sentinel)".into());
        }
        super::execution::validate_threads(self.threads)?;
        super::execution::validate_window(self.window_size)?;
        super::resources::validate_memory_budget(self.memory_budget_gib)?;
        let graph = &self.graph;
        let storage = self
            .storage
            .unwrap_or_else(|| VectorStorageKind::for_input(format));
        let mut policy = graph.source;
        if graph.staged_file.is_some() {
            if matches!(
                policy,
                GraphSourcePolicy::Rust | GraphSourcePolicy::Supplied
            ) {
                return Err("staged_file conflicts with graph source rust or supplied".into());
            }
            policy = GraphSourcePolicy::Parlayann;
        }
        let source = policy.candidate();
        let defaults = if storage == VectorStorageKind::U8 {
            Cascade::native_u8(self.metric)
        } else {
            Cascade::default_for_specified_dimension_and_metric(dimension, self.metric)
        };
        let cascade = self.cascade.apply(defaults)?;
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
            #[cfg(feature = "benchmark")]
            benchmark_trust_cache: false,
            #[cfg(feature = "benchmark")]
            benchmark_cache_file: None,
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
            auto_pin: self.auto_pin,
            storage_options: self.storage_options.clone(),
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
        if config.graph_source != super::GraphSource::Supplied {
            config.build_config()?.validate()?;
        }
        config.cache_namespace = config.compute_cache_namespace();
        Ok(config)
    }
}
