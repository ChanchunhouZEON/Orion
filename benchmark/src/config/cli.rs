/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resolve benchmark configuration before loading vectors or constructing an index.
//!
//! This module combines:
//!
//! - dataset presets from YAML,
//! - legacy environment variables,
//! - explicit command-line overrides,
//! - dimension- and metric-dependent cascade defaults,
//!
//! into a validated [`ResolvedRunConfig`].
//!
//! Resolution is intentionally separated from dataset loading and index construction so
//! configuration errors can be reported before large allocations or expensive graph work
//! begin.
//!
//! The resulting configuration also derives the graph-cache identity used to distinguish
//! graphs built from different datasets or construction parameters.
use crate::config::presets::{
    try_load_root, try_resolve_placeholders, RawBaseGraph, RawOrionOverride, RawRoot,
    SweepOverrides,
};
use clap::Parser;
use orion_ann::benchmark::config::{
    AdmissionChoice, Cascade, PrefilterChoice, RerankChoice, SearchMetric,
};
use orion_ann::execution::VectorFormat;
use orion_ann::{Error, Result};

use orion_ann::benchmark::ResolvedRunConfig;
use std::path::PathBuf;

use orion_ann::benchmark::config::{
    GraphSource, GraphSourcePolicy, PreflightFormat, VectorStorageKind,
};

/// Command-line configuration accepted by the Orion benchmark executable.
///
/// Most fields are optional because their effective values may come from a dataset
/// preset or global YAML defaults. Explicit CLI values take precedence during
/// [`resolve`](Self::resolve).
///
/// `Args` represents unresolved user input. Code that loads data or constructs an
/// index should normally operate on [`ResolvedRunConfig`] instead.
#[derive(Debug, Clone, Parser)]
#[command(name = "orion", about = "Shared Orion input and search configuration")]
pub struct Args {
    /// Optional dataset preset name defined in the sweep configuration.
    ///
    /// Omit the preset (or use `anonymous`) to resolve defaults from metric and dimension.
    pub dataset: Option<String>,
    /// Legacy positional limit retained for compatibility with older benchmark scripts.
    ///
    /// Prefer [`max_points`](Self::max_points) in new invocations.
    pub legacy_max_points: Option<usize>,
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub base: Option<PathBuf>,
    #[arg(long)]
    pub query: Option<PathBuf>,
    /// Maximum query vectors buffered from a stream per search batch.
    #[arg(long, default_value_t = 1024)]
    pub query_batch_size: usize,
    #[arg(long)]
    pub groundtruth: Option<PathBuf>,
    /// Disable preset ground truth and report search metrics without recall.
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
    /// Estimate memory from preset metadata or available headers; no payload loading.
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
    /// Build/import and save graph cache, then exit before calibration/search.
    #[arg(long)]
    pub prepare_only: bool,
    /// Physical vector dimension. Inferred from the selected base format if omitted.
    #[arg(long)]
    pub dimension: Option<usize>,
    #[arg(long, value_enum)]
    pub metric: Option<SearchMetric>,
    #[arg(long)]
    pub prefilter: Option<PrefilterChoice>,
    #[arg(long)]
    pub admission: Option<AdmissionChoice>,
    #[arg(long)]
    pub rerank: Option<RerankChoice>,
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
    /// Default auto: PA cache/export first, then Rust (f32 only). Explicit sources are strict.
    #[arg(long, value_enum)]
    pub graph_source: Option<GraphSourcePolicy>,
    #[arg(long)]
    pub staged_file: Option<PathBuf>,
    /// Separate cache location for an alternate graph built from the same vectors.
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
    /// Trust an existing .bin/.pgraph pair or .cache.json record (no SHA-256 scan).
    /// The experiment owner must match the base and graph; layout checks remain.
    #[arg(long)]
    pub cache_file: Option<PathBuf>,
    #[arg(long, conflicts_with = "legacy_max_points")]
    pub max_points: Option<usize>,
    #[arg(long)]
    pub k: Option<usize>,
    #[command(flatten)]
    pub sweep: SweepOverrides,
    #[arg(long)]
    pub print_config: bool,
}

/// Compatibility settings read from legacy environment variables.
///
/// These values preserve older shell-based benchmark workflows and participate
/// in configuration resolution only when no higher-precedence CLI value is present.
#[derive(Default)]
struct LegacyEnvironment {
    graph_source: Option<GraphSourcePolicy>,
    staged_file: Option<PathBuf>,
    pa_root: Option<PathBuf>,
    local_pct: Option<usize>,
    profiling: bool,
}

impl LegacyEnvironment {
    /// Reads all supported legacy environment variables.
    ///
    /// Missing variables are treated as unspecified. Malformed graph-source values
    /// are reported as configuration errors.
    fn read() -> Result<Self> {
        Ok(Self {
            graph_source: std::env::var("ORION_GRAPH")
                .ok()
                .map(|s| parse_graph_source(&s))
                .transpose()?,
            staged_file: std::env::var_os("ORION_STAGED_FILE").map(PathBuf::from),
            pa_root: std::env::var_os("PA_ROOT").map(PathBuf::from),
            local_pct: std::env::var("ORION_LOCAL_PCT")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|v| (1..=100).contains(v)),
            profiling: std::env::var_os("ORION_PROFILE_MARKER").is_some(),
        })
    }
}

/// Parses a legacy graph-source spelling accepted outside Clap's `ValueEnum`.
fn parse_graph_source(s: &str) -> Result<GraphSourcePolicy> {
    match s.to_ascii_lowercase().as_str() {
        "auto" => Ok(GraphSourcePolicy::Auto),
        "rust" => Ok(GraphSourcePolicy::Rust),
        "pa" | "parlayann" => Ok(GraphSourcePolicy::Parlayann),
        _ => Err(Error::from(format!(
            "unknown graph source {s:?}; expected auto, rust or parlayann"
        ))),
    }
}

impl Args {
    /// Resolves this CLI input into a complete runtime configuration.
    ///
    /// Resolution merges, in precedence order, explicit CLI values, legacy environment
    /// settings, dataset-specific YAML overrides, and global defaults.
    ///
    /// An available file base is hashed for cache lookup. An absent preset base can still be used for offline budgeting.
    ///
    /// # Errors
    ///
    /// Returns an error if the custom configuration file cannot be read or parsed,
    /// a referenced dataset preset does not exist, required settings are missing,
    /// or any resolved option combination is invalid.
    pub fn resolve(&self) -> Result<ResolvedRunConfig> {
        let custom_root: Option<RawRoot> = self
            .config
            .as_ref()
            .map(|path| {
                let text = std::fs::read_to_string(path).map_err(|e| {
                    let message = format!("{}: {e}", path.display());
                    Error::from(e).with_message(message)
                })?;
                serde_yaml::from_str(&text).map_err(|e| {
                    Error::invalid_data(format!("invalid YAML: {e}"))
                        .at("read_config")
                        .with_detail("path", path.to_string_lossy().into_owned())
                        .with_source(e)
                })
            })
            .transpose()?;
        self.resolve_with(
            match custom_root.as_ref() {
                Some(root) => root,
                None => try_load_root()?,
            },
            &LegacyEnvironment::read()?,
        )
    }

    /// Experimental sweeps materialize queries so trials can replay identical inputs.
    pub fn resolve_for_sweep(&self) -> Result<ResolvedRunConfig> {
        let mut config = self.resolve()?;
        config.replay_queries = true;
        Ok(config)
    }

    /// Resolves configuration against an explicit preset root and legacy environment.
    ///
    /// This helper exists primarily so configuration resolution can be tested without
    /// reading process-global environment state.
    ///
    /// Resolution broadly proceeds as:
    ///
    /// 1. select the dataset preset,
    /// 2. resolve paths and file formats,
    /// 3. resolve metric, dimension, and resident storage,
    /// 4. resolve cascade and graph-construction settings,
    /// 5. validate cross-option constraints,
    /// 6. resolve graph source and staged export,
    /// 7. resolve sweep parameters,
    /// 8. derive cache compatibility and cache identity.
    fn resolve_with(&self, root: &RawRoot, env: &LegacyEnvironment) -> Result<ResolvedRunConfig> {
        // A preset is optional. Anonymous runs still use metric/dimension defaults.
        let preset_name = self.dataset.as_deref().filter(|name| *name != "anonymous");
        let dataset = preset_name.unwrap_or("anonymous").to_owned();
        let preset = preset_name
            .map(|name| {
                root.datasets.get(name).ok_or_else(|| {
                    format!(
                        "unknown dataset shortcut {name:?}; omit the name for an anonymous dataset"
                    )
                })
            })
            .transpose()?;
        let paths = preset.and_then(|p| p.paths.as_ref());

        // Explicit paths override only the corresponding preset path.
        let required_path = |cli: &Option<PathBuf>, default: Option<&String>, flag: &str| {
            cli.clone()
                .or_else(|| default.map(PathBuf::from))
                .ok_or_else(|| format!("--{flag} is required without a dataset preset"))
        };
        let base = required_path(&self.base, paths.map(|p| &p.base), "base")?;
        let query = self
            .query
            .clone()
            .or_else(|| paths.map(|p| PathBuf::from(&p.query)));
        if query.is_none() && !self.prepare_only {
            return Err(
                "--query is required for search (use --prepare-only to build without queries)"
                    .into(),
            );
        }
        let groundtruth = if self.no_groundtruth {
            None
        } else {
            self.groundtruth.clone().or_else(|| {
                paths
                    .and_then(|p| p.groundtruth.as_ref())
                    .map(PathBuf::from)
            })
        };

        let base_format = self
            .base_format
            .map(Ok)
            .unwrap_or_else(|| VectorFormat::infer(&base))?;
        let query_format = self.query_format.map(Ok).unwrap_or_else(|| {
            query
                .as_deref()
                .map(VectorFormat::infer)
                .unwrap_or(Ok(base_format))
        })?;
        orion_ann::benchmark::config::validate_memory_budget(self.memory_budget_gib)?;

        if self.query_batch_size == 0 {
            return Err("--query-batch-size must be positive".into());
        }

        let metric = self
            .metric
            .or_else(|| preset.and_then(|p| p.metric))
            .ok_or(
                "--metric is required; do not infer a dataset objective from its admission kernel",
            )?;
        let dimension = match self.dimension.or_else(|| preset.map(|p| p.dimension)) {
            Some(d) => d,
            None => orion_ann::execution::inspect_vector_file(&base, base_format)?.dimension,
        };

        let defaults = &root.defaults.orion;
        let dataset_overrides = preset.and_then(|p| p.orion.as_ref());
        let vector_storage = match self.vector_storage {
            Some(storage) => storage,
            None => match dataset_overrides.and_then(|o| o.vector_storage.as_deref()) {
                None | Some("f32") => VectorStorageKind::F32,
                Some("u8") => VectorStorageKind::U8,
                Some(other) => {
                    return Err(Error::from(format!(
                        "unknown vector_storage {other:?}; expected f32 or u8"
                    )));
                }
            },
        };
        let cascade = self.resolve_cascade(dimension, metric, dataset_overrides)?;

        // Construction settings use CLI > dataset override > global default.
        let alpha = self
            .alpha
            .or_else(|| dataset_overrides.and_then(|o| o.alpha))
            .unwrap_or(defaults.alpha);
        let graph_degree = self
            .graph_degree
            .or_else(|| dataset_overrides.and_then(|o| o.graph_degree))
            .unwrap_or(defaults.graph_degree);
        let build_l = self
            .build_l
            .or_else(|| dataset_overrides.and_then(|o| o.build_search_list_size))
            .unwrap_or(defaults.build_search_list_size);
        let max_extra = self
            .max_extra
            .or_else(|| dataset_overrides.and_then(|o| o.max_extra))
            .unwrap_or(defaults.max_extra);
        let window_size = self
            .window_size
            .or_else(|| dataset_overrides.and_then(|o| o.window_size))
            .unwrap_or(defaults.window_size);
        let max_points = self
            .max_points
            .or(self.legacy_max_points)
            .or_else(|| preset.and_then(|p| p.max_points))
            .unwrap_or(usize::MAX);

        // Historical imported/cache recipes may use alpha < 1. Validate the
        // actual builder recipe only on a cache miss that requires construction.
        if !alpha.is_finite()
            || alpha <= 0.0
            || graph_degree == 0
            || build_l == 0
            || build_l > u32::MAX as usize
        {
            return Err(
                "alpha, graph-degree and build-l must be positive and build-l must fit u32".into(),
            );
        }
        orion_ann::benchmark::config::validate_window(window_size)?;
        if max_points == 0 {
            return Err("max-points must be positive".into());
        }

        let (graph_source_policy, staged_file, staged_file_explicit) = self.resolve_graph_source(
            preset.and_then(|dataset| dataset.base_graph.as_ref()),
            env,
            max_extra,
        )?;

        // Auto starts with a PA candidate. Resolve its cache identity before
        // probing availability; policy itself must never enter the cache key.
        let graph_source = graph_source_policy.candidate();
        let index_plan = orion_ann::benchmark::config::IndexPlan::resolve(
            vector_storage,
            dimension,
            metric,
            base_format,
            query_format,
            cascade,
            graph_source,
        )?;

        // Profiling supplies a trial count only when the CLI did not specify one.
        let mut overrides = self.sweep.clone();
        if overrides.trials.is_none() && env.profiling {
            overrides.trials = root
                .defaults
                .sweep
                .profiles
                .get("profiling")
                .and_then(|p| p.trials);
        }
        let sweep = root.defaults.sweep.resolve(
            "sweep",
            self.k.unwrap_or(root.defaults.sweep.k),
            &overrides,
        )?;

        orion_ann::benchmark::config::validate_threads(sweep.threads)?;

        let num_points = if self.base.is_none() {
            preset.and_then(|p| p.num_points)
        } else {
            None
        };
        let num_queries = if self.query.is_none() {
            preset.and_then(|p| p.num_queries)
        } else {
            None
        };
        if num_points == Some(0) || num_queries == Some(0) {
            return Err("preset num_points and num_queries must be positive when provided".into());
        }

        let mut run = ResolvedRunConfig {
            storage_options: Default::default(),
            benchmark_trust_cache: true,
            benchmark_cache_file: self.cache_file.clone(),
            base_content_id: None,
            dataset,
            base,
            query,
            groundtruth,
            num_points,
            num_queries,
            query_batch_size: self.query_batch_size,
            replay_queries: false,
            base_format,
            query_format,
            index_plan,
            memory_budget_gib: self.memory_budget_gib,
            auto_pin: !self.no_auto_pin,
            prepare_only: self.prepare_only,
            dimension,
            metric,
            alpha,
            graph_degree,
            build_l,
            max_extra,
            window_size,
            graph_source,
            graph_source_policy,
            graph_selection_reason: "explicit source",
            staged_file,
            max_points,
            sweep,
            cache_namespace: String::new(),
            cache_dir: self.cache_dir.clone().unwrap_or_else(|| {
                PathBuf::from(if graph_source == GraphSource::Parlayann {
                    "cache/orion_parlayann"
                } else {
                    "cache/orion"
                })
            }),
            staged_file_explicit,
            local_pct: env.local_pct.unwrap_or(60),
        };

        // Default CAS lookup still needs the base content identity. Explicit cache
        // selection bypasses that scan as well as payload authentication.
        if self.cache_file.is_none() {
            use sha2::Digest;
            match std::fs::File::open(&run.base) {
                Ok(mut file) => {
                    let mut hash = sha2::Sha256::new();
                    std::io::copy(&mut file, &mut hash)?;
                    run.base_content_id = Some(format!("file-{:x}", hash.finalize()));
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(Error::io("identify base", &run.base, e)),
            }
        }
        run.cache_namespace = run.compute_cache_namespace();
        if graph_source_policy == GraphSourcePolicy::Auto {
            run.select_automatic_graph(self.cache_dir.is_some())?;
        }

        Ok(run)
    }

    /// Resolves the effective search cascade.
    ///
    /// The dimension/metric selector provides the initial cascade. Dataset-specific
    /// axis overrides are then applied, followed by explicit CLI overrides.
    ///
    /// Global Orion defaults are intentionally not consulted here so previously unseen
    /// datasets can select a cascade from their dimension and metric alone.
    ///
    /// # Errors
    ///
    /// Returns an error when the resulting axis combination is invalid for `metric`.
    fn resolve_cascade(
        &self,
        dimension: usize,
        metric: SearchMetric,
        dataset_overrides: Option<&RawOrionOverride>,
    ) -> Result<Cascade> {
        let mut cascade = Cascade::default_for_specified_dimension_and_metric(dimension, metric);
        if let Some(o) = dataset_overrides {
            cascade = cascade.with_named_overrides(
                o.prefilter.as_deref(),
                o.admission.as_deref(),
                o.rerank.as_deref(),
            )?;
        }
        // Explicit axes win over the dataset preset; final validation is in IndexPlan.
        let cascade = cascade.with_overrides(self.prefilter, self.admission, self.rerank);

        Ok(cascade)
    }

    /// CLI > legacy environment > YAML > auto. An explicitly supplied staged
    /// path requests PA strictly, including alongside `--graph-source auto`.
    /// Header/cache probing happens after the candidate configuration is complete.
    fn resolve_graph_source(
        &self,
        preset_graph: Option<&RawBaseGraph>,
        env: &LegacyEnvironment,
        max_extra: usize,
    ) -> Result<(GraphSourcePolicy, Option<PathBuf>, bool)> {
        let explicit_source = self
            .graph_source
            .or_else(|| {
                self.staged_file
                    .as_ref()
                    .map(|_| GraphSourcePolicy::Parlayann)
            })
            .or(env.graph_source)
            .or_else(|| {
                env.staged_file
                    .as_ref()
                    .map(|_| GraphSourcePolicy::Parlayann)
            });
        let mut graph_source = explicit_source.unwrap_or(
            preset_graph
                .and_then(|b| b.source.as_deref())
                .map(parse_graph_source)
                .transpose()?
                .unwrap_or(GraphSourcePolicy::Auto),
        );

        let staged_file_explicit = self.staged_file.is_some() || env.staged_file.is_some();
        let pa_root = env
            .pa_root
            .as_deref()
            .unwrap_or(std::path::Path::new("../ParlayANN"));
        let staged_file = self
            .staged_file
            .clone()
            .or_else(|| env.staged_file.clone())
            .or_else(|| {
                preset_graph
                    .and_then(|b| b.staged_file.as_ref())
                    .and_then(|s| {
                        try_resolve_placeholders(
                            &s.replace("${max_extra}", &max_extra.to_string())
                                .replace("${PA_ROOT}", &pa_root.to_string_lossy()),
                        )
                    })
                    .map(PathBuf::from)
            });

        if graph_source == GraphSourcePolicy::Rust && self.staged_file.is_some() {
            return Err("--staged-file conflicts with --graph-source rust".into());
        }
        if graph_source == GraphSourcePolicy::Auto && staged_file_explicit {
            graph_source = GraphSourcePolicy::Parlayann;
        }
        Ok((graph_source, staged_file, staged_file_explicit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn root() -> RawRoot {
        serde_yaml::from_str(crate::config::presets::DEFAULT_CONFIG).unwrap()
    }
    fn resolve(args: &[&str]) -> Result<ResolvedRunConfig> {
        Args::try_parse_from(args)
            .map_err(|e| Error::invalid_argument(e.to_string()).with_source(e))?
            .resolve_with(&root(), &LegacyEnvironment::default())
    }

    #[test]
    fn benchmark_cache_selection_and_auto_pin_are_independent() {
        let selected = resolve(&[
            "orion",
            "sift",
            "--graph-source",
            "parlayann",
            "--cache-file",
            "old.bin",
        ])
        .unwrap();
        assert!(selected.benchmark_trust_cache);
        assert!(selected.auto_pin);
        assert_eq!(
            selected.benchmark_cache_file,
            Some(PathBuf::from("old.bin"))
        );
        assert!(selected.base_content_id.is_none());
        let disabled = resolve(&[
            "orion",
            "sift",
            "--graph-source",
            "parlayann",
            "--no-auto-pin",
            "--cache-file",
            "old.bin",
        ])
        .unwrap();
        assert!(!disabled.auto_pin);
        assert!(disabled.benchmark_trust_cache);
    }

    #[test]
    fn large_sift_presets_select_native_bytes_and_exact_prefixes() {
        for (name, count, groundtruth) in [
            ("sift10m", 10_000_000, "groundtruth.10M.ivecs"),
            ("sift100m", 100_000_000, "idx_100M.ivecs"),
            ("sift1b", 1_000_000_000, "idx_1000M.ivecs"),
        ] {
            let config = resolve(&["orion", name]).unwrap();
            assert_eq!(config.storage(), VectorStorageKind::U8);
            assert_eq!(config.max_points, count);
            assert_eq!(config.base_format, VectorFormat::Bvecs);
            assert_eq!(config.query_format, VectorFormat::Bvecs);
            assert!(config.groundtruth.as_ref().unwrap().ends_with(groundtruth));
            assert_eq!(config.cascade().prefilter, PrefilterChoice::None);
            assert_eq!(config.cascade().admission, AdmissionChoice::NativeL2U8);
            assert_eq!(config.cascade().rerank, RerankChoice::None);
            assert_eq!(config.graph_source, GraphSource::Parlayann);
            assert!(config.staged_file.unwrap().to_string_lossy().contains(name));
        }
        assert_eq!(
            resolve(&["orion", "sift10m", "--max-points", "1000"])
                .unwrap()
                .max_points,
            1000
        );
    }

    #[test]
    fn native_storage_is_validated_and_has_a_distinct_cache_identity() {
        let native = resolve(&["orion", "sift10m"]).unwrap();
        let float = resolve(&[
            "orion",
            "sift10m",
            "--vector-storage",
            "f32",
            "--admission",
            "l2-u8",
        ])
        .unwrap();
        assert_ne!(native.cache_namespace, float.cache_namespace);
        assert_eq!(native.storage(), VectorStorageKind::U8);
        assert_eq!(native.cascade(), native.cascade());
        assert_eq!(float.cascade(), float.cascade());
        assert!(resolve(&["orion", "sift10m", "--vector-storage", "f32"])
            .unwrap_err()
            .to_string()
            .contains("native-l2-u8 requires --vector-storage u8"));
        assert!(resolve(&["orion", "sift10m", "--admission", "l2-u8"])
            .unwrap_err()
            .to_string()
            .contains("--admission native-l2-u8"));
        assert_eq!(native.storage(), VectorStorageKind::U8);
        assert_eq!(float.storage(), VectorStorageKind::F32);

        assert!(resolve(&["orion", "sift10m", "--rerank", "f32"]).is_err());
        assert!(resolve(&["orion", "sift10m", "--admission", "l2-u16"]).is_err());
        assert!(resolve(&["orion", "sift10m", "--prefilter", "jl"]).is_err());
        assert!(resolve(&["orion", "sift10m", "--query", "query.fvecs"]).is_err());
        assert!(resolve(&["orion", "sift10m", "--base", "base.fbin"]).is_err());
        assert!(resolve(&["orion", "sift10m", "--graph-source", "rust"]).is_err());
        assert!(resolve(&["orion", "sift10m", "--dimension", "33"]).is_ok());
    }

    #[test]
    fn preserves_all_named_cascades_and_build_recipes() {
        for (name, d, a, r, l, pre, admission, rerank) in [
            (
                "sift",
                128,
                1.15,
                64,
                128,
                PrefilterChoice::None,
                AdmissionChoice::L2U8,
                RerankChoice::F32,
            ),
            (
                "deep10m",
                128,
                1.05,
                64,
                128,
                PrefilterChoice::None,
                AdmissionChoice::L2U8,
                RerankChoice::F32,
            ),
            (
                "glove25",
                32,
                1.0,
                100,
                200,
                PrefilterChoice::None,
                AdmissionChoice::MipsI8,
                RerankChoice::IpF32,
            ),
            (
                "glove100",
                100,
                1.0,
                100,
                200,
                PrefilterChoice::None,
                AdmissionChoice::MipsI8,
                RerankChoice::IpF32,
            ),
            (
                "gist",
                960,
                1.1,
                100,
                200,
                PrefilterChoice::Jl,
                AdmissionChoice::L2Kt,
                RerankChoice::F32,
            ),
            (
                "fashion-mnist",
                784,
                1.1,
                40,
                80,
                PrefilterChoice::None,
                AdmissionChoice::L2Kt,
                RerankChoice::F32,
            ),
            (
                "msmarco_bert_1M",
                768,
                1.0,
                64,
                128,
                PrefilterChoice::None,
                AdmissionChoice::MipsI8,
                RerankChoice::IpF32,
            ),
            (
                "wiki_ada_1M",
                1536,
                0.98,
                64,
                128,
                PrefilterChoice::Jl,
                AdmissionChoice::MipsI8,
                RerankChoice::IpF32,
            ),
        ] {
            let c = resolve(&["orion", name, "--graph-source", "rust"]).unwrap();
            assert_eq!(
                (c.dimension, c.alpha, c.graph_degree, c.build_l),
                (d, a, r, l),
                "{name}"
            );
            assert_eq!(
                c.cascade(),
                Cascade {
                    prefilter: pre,
                    admission,
                    rerank
                },
                "{name}"
            );
        }
    }

    #[test]
    fn custom_inputs_need_no_dataset_name() {
        let c = resolve(&[
            "orion",
            "--base",
            "new.fvecs",
            "--query",
            "q.fvecs",
            "--groundtruth",
            "gt.ivecs",
            "--dimension",
            "960",
            "--metric",
            "l2",
            "--k",
            "100",
        ])
        .unwrap();
        assert_eq!(c.dataset, "anonymous");
        assert_eq!(c.cascade().admission, AdmissionChoice::L2Kt);
        assert_eq!(c.sweep.calibration_l, 200);
        assert_eq!(c.graph_source, GraphSource::Parlayann);
    }

    #[test]
    fn overrides_are_independent_and_change_cache_identity() {
        let a = resolve(&["orion", "sift", "--graph-source", "rust"]).unwrap();
        let b = resolve(&[
            "orion",
            "sift",
            "--graph-source",
            "rust",
            "--base",
            "new.fvecs",
            "--alpha",
            "1.3",
            "--prefilter",
            "jl",
        ])
        .unwrap();
        assert_eq!(b.alpha, 1.3);
        assert_eq!(b.cascade().prefilter, PrefilterChoice::Jl);
        assert_eq!(b.cascade().admission, a.cascade().admission);
        assert_ne!(a.cache_namespace, b.cache_namespace);
        let c = resolve(&["orion", "sift", "--graph-source", "rust", "--k", "100"]).unwrap();
        assert_eq!(a.cache_namespace, c.cache_namespace);
    }

    #[test]
    fn invalid_inputs_fail_before_execution() {
        for args in [
            vec!["orion", "missing"],
            vec!["orion", "sift", "--dimension", "0"],
            vec!["orion", "sift", "--metric", "inner-product"],
            vec!["orion", "sift", "--window-size", "0"],
            vec!["orion", "sift", "--k", "100", "--ls", "16"],
            vec![
                "orion",
                "--base",
                "b",
                "--query",
                "q",
                "--groundtruth",
                "g",
                "--dimension",
                "128",
            ],
        ] {
            assert!(resolve(&args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn yaml_entry_without_axes_uses_selector() {
        let mut root = root();
        root.datasets.insert(
            "new".into(),
            serde_yaml::from_str(
                "dimension: 960\nmetric: l2\npaths: {base: b, query: q, groundtruth: g}",
            )
            .unwrap(),
        );
        let c = Args::try_parse_from(["orion", "new"])
            .unwrap()
            .resolve_with(&root, &LegacyEnvironment::default())
            .unwrap();
        assert_eq!(
            c.cascade(),
            Cascade::default_for_specified_dimension_and_metric(960, SearchMetric::L2)
        );
    }

    #[test]
    fn cli_graph_source_overrides_legacy_environment() {
        let args = Args::try_parse_from(["orion", "sift", "--graph-source", "rust"]).unwrap();
        let env = LegacyEnvironment {
            graph_source: Some(GraphSourcePolicy::Parlayann),
            staged_file: Some("import.staged".into()),
            ..Default::default()
        };
        let c = args.resolve_with(&root(), &env).unwrap();
        assert_eq!(c.graph_source, GraphSource::Rust);
        let args =
            Args::try_parse_from(["orion", "sift", "--staged-file", "explicit.staged"]).unwrap();
        let c = args.resolve_with(&root(), &env).unwrap();
        assert_eq!(c.graph_source, GraphSource::Parlayann);
        assert_eq!(c.staged_file, Some("explicit.staged".into()));
    }

    #[test]
    fn cache_identity_includes_metric_and_exact_alpha() {
        let mut c = resolve(&["orion", "sift", "--graph-source", "rust"]).unwrap();
        let original = c.compute_cache_namespace();
        c.alpha += 0.001;
        assert_ne!(original, c.compute_cache_namespace());
        c.metric = SearchMetric::InnerProduct;
        let ip = c.compute_cache_namespace();
        c.metric = SearchMetric::Cosine;
        assert_ne!(ip, c.compute_cache_namespace());
    }

    #[test]
    fn pa_root_is_optional_and_does_not_identify_the_cache() {
        let args = Args::try_parse_from(["orion", "sift", "--graph-source", "parlayann"]).unwrap();
        let default = args
            .resolve_with(&root(), &LegacyEnvironment::default())
            .unwrap();
        assert_eq!(default.graph_source, GraphSource::Parlayann);
        assert!(default
            .staged_file
            .as_ref()
            .unwrap()
            .starts_with("../ParlayANN"));
        let elsewhere = args
            .resolve_with(
                &root(),
                &LegacyEnvironment {
                    pa_root: Some("/different/checkout".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_ne!(default.staged_file, elsewhere.staged_file);
        assert_eq!(default.cache_namespace, elsewhere.cache_namespace);
    }

    #[test]
    fn custom_pa_run_can_resolve_without_an_export() {
        let args = Args::try_parse_from([
            "orion",
            "--base",
            "new.fvecs",
            "--query",
            "q.fvecs",
            "--groundtruth",
            "gt.ivecs",
            "--dimension",
            "32",
            "--metric",
            "l2",
            "--graph-source",
            "parlayann",
        ])
        .unwrap();
        let config = args
            .resolve_with(&root(), &LegacyEnvironment::default())
            .unwrap();
        assert_eq!(config.graph_source, GraphSource::Parlayann);
        assert!(config.staged_file.is_none());
    }

    #[test]
    fn graph_recipe_has_a_full_sha256_name() {
        let config = resolve(&["orion", "sift"]).unwrap();
        assert_eq!(config.cache_namespace.len(), 64);
        assert!(config
            .cache_namespace
            .bytes()
            .all(|b| b.is_ascii_hexdigit()));
    }
}
