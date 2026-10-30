//! Programmatic configuration. These defaults do not read YAML or environment overrides.
use crate::{
    config::plan::SearchPlan,
    config::ResolvedSweep,
    config::*,
    data::VectorFormat,
    search::cascade::{AdmissionChoice, Cascade, PrefilterChoice, RerankChoice, SearchMetric},
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GraphConfig {
    pub source: GraphSourcePolicy,
    pub staged_file: Option<PathBuf>,
    pub degree: u32,
    pub build_l: usize,
    pub alpha: f32,
    pub max_extra: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            source: GraphSourcePolicy::Auto,
            staged_file: None,
            degree: 64,
            build_l: 100,
            alpha: 1.2,
            max_extra: 16,
        }
    }
}

/// Overrides use the same vocabulary and validation as the CLI cascade.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CascadeConfig {
    pub prefilter: Option<String>,
    pub admission: Option<String>,
    pub rerank: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IndexOptions {
    pub metric: SearchMetric,
    pub storage: Option<VectorStorageKind>,
    pub base_format: Option<VectorFormat>,
    pub graph: GraphConfig,
    pub cascade: CascadeConfig,
    pub cache_dir: PathBuf,
    pub threads: usize,
    pub window_size: usize,
    pub memory_budget_gib: Option<f64>,
    pub max_points: Option<usize>,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            metric: SearchMetric::L2,
            storage: None,
            base_format: None,
            graph: GraphConfig::default(),
            cascade: CascadeConfig::default(),
            cache_dir: "cache/orion-py".into(),
            threads: 8,
            window_size: 5,
            memory_budget_gib: None,
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
    ) -> Result<ResolvedRunConfig, String> {
        vector::PaddedLayout::for_kernel::<vector::L2F32Distance>(dimension)?;
        if count == 0 || count >= u32::MAX as usize || self.max_points == Some(0) {
            return Err("base must contain 1..u32::MAX points (excluding the sentinel)".into());
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
        if graph.degree == 0
            || graph.degree > u32::MAX - 4
            || graph.build_l < graph.degree as usize
            || graph.build_l > u32::MAX as usize
            || !graph.alpha.is_finite()
            || graph.alpha < 1.0
            || graph.max_extra > u32::MAX as usize - graph.degree as usize - 4
        {
            return Err("invalid graph parameters: require degree > 0, build_l >= degree, finite alpha >= 1, and u32-sized graph fields".into());
        }
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
        let search_plan = SearchPlan::resolve(
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
            search_plan,
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
                profile: "runtime".into(),
                search_list_sizes: vec![64],
                threads: self.threads,
                trials: 1,
                calibration_samples: 200,
                calibration_l: 48,
            },
            cache_namespace: String::new(),
            cache_dir: self.cache_dir.clone(),
            staged_file_explicit: graph.staged_file.is_some(),
            legacy_dataset: None,
            legacy_local_pct: 60,
        };
        config.cache_namespace = config.compute_cache_namespace();
        if policy == GraphSourcePolicy::Auto {
            config.select_automatic_graph(true)?;
        }
        Ok(config)
    }
}
