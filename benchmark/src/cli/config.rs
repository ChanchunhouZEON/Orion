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
use super::data::VectorFormat;
use crate::cascade::{AdmissionChoice, Cascade, PrefilterChoice, RerankChoice, SearchMetric};
use crate::config::{
    load_root, try_resolve_placeholders, RawBaseGraph, RawOrionOverride, RawRoot, ResolvedSweep,
    SweepOverrides,
};
use clap::Parser;
use serde::Serialize;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

/// Source used to obtain the base proximity graph.
///
/// The graph can either be constructed by DiskANN native Rust builder or imported
/// from a ParlayANN-generated staged graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum GraphSource {
    /// Construct the graph using DiskANN native Rust implementation with extra
    /// candidates complemented.
    Rust,

    /// Import a graph produced by ParlayANN.
    #[value(alias = "pa")]
    Parlayann,
}

/// Element representation used for resident base vectors.
///
/// This is independent of the on-disk [`VectorFormat`]. For example, vectors read
/// from a `.bvecs` file may either remain as native `u8` values or be expanded into
/// `f32` values after loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum VectorStorageKind {
    /// Store each coordinate as `f32`.
    F32,

    /// Keep each coordinate in its native unsigned 8-bit representation if the base vector
    /// is in `u8` stored form.
    U8,
}

/// Command-line configuration accepted by the Orion benchmark executable.
///
/// Most fields are optional because their effective values may come from a dataset
/// preset or global YAML defaults. Explicit CLI values take precedence during
/// [`resolve`](Self::resolve).
///
/// `Args` represents unresolved user input. Code that loads data or constructs an
/// index should normally operate on [`ResolvedRunConfig`] instead.
#[derive(Debug, Parser)]
#[command(name = "orion", about = "Configurable Orion QPS-recall sweep")]
pub struct Args {
    /// Optional dataset preset name defined in the sweep configuration.
    ///
    /// When omitted together with all explicit dataset paths, the bundled `sift`
    /// preset is selected for backward compatibility.
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
    #[arg(long)]
    pub groundtruth: Option<PathBuf>,
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
    /// Print input and resident-memory lower bounds without loading base or graph.
    #[arg(long)]
    pub preflight: bool,
    /// Reject configurations whose known resident components exceed this budget.
    ///
    /// The value is expressed in GiB and is only a lower-bound check; temporary
    /// construction allocations are not guaranteed to fit within this limit.
    #[arg(long)]
    pub memory_budget_gib: Option<f64>,
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
    #[arg(long, value_enum)]
    pub graph_source: Option<GraphSource>,
    #[arg(long)]
    pub staged_file: Option<PathBuf>,
    /// Separate cache location for an alternate graph built from the same vectors.
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
    #[arg(long, conflicts_with = "legacy_max_points")]
    pub max_points: Option<usize>,
    #[arg(long)]
    pub k: Option<usize>,
    #[command(flatten)]
    pub sweep: SweepOverrides,
    #[arg(long)]
    pub print_config: bool,
}

/// Fully resolved and validated configuration for one Orion benchmark run.
///
/// Unlike [`Args`], all required settings have concrete values. The resolution
/// process has already applied dataset presets, legacy compatibility settings,
/// global defaults, and explicit CLI overrides.
///
/// Instances of this type are intended to be passed to the data-loading, graph
/// preparation, calibration, and search stages.
///
/// # Cache identity
///
/// [`cache_namespace`](Self::cache_namespace) identifies graph state derived from
/// the effective base dataset and graph-construction parameters. Query-only
/// settings intentionally do not participate in that identity.
#[derive(Debug, Serialize)]
pub struct ResolvedRunConfig {
    /// Logical dataset preset name, if this run originated from one.
    pub dataset: Option<String>,
    pub base: PathBuf,
    pub query: PathBuf,
    pub groundtruth: PathBuf,
    pub base_format: VectorFormat,
    pub query_format: VectorFormat,
    pub vector_storage: VectorStorageKind,
    pub memory_budget_gib: Option<f64>,
    pub prepare_only: bool,
    pub dimension: usize,
    pub metric: SearchMetric,
    pub cascade: Cascade,
    pub alpha: f32,
    pub graph_degree: u32,
    pub build_l: usize,
    pub max_extra: usize,
    pub window_size: usize,
    pub graph_source: GraphSource,
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
    /// Matching legacy bundled preset, when this configuration is compatible with an
    /// older cache naming scheme.
    #[serde(skip)]
    pub legacy_dataset: Option<String>,
    #[serde(skip)]
    pub legacy_local_pct: usize,
}

/// Compatibility settings read from legacy environment variables.
///
/// These values preserve older shell-based benchmark workflows and participate
/// in configuration resolution only when no higher-precedence CLI value is present.
#[derive(Default)]
struct LegacyEnvironment {
    graph_source: Option<GraphSource>,
    staged_file: Option<PathBuf>,
    pa_root: Option<PathBuf>,
    legacy_local_pct: Option<usize>,
    profiling: bool,
}

impl LegacyEnvironment {
    /// Reads all supported legacy environment variables.
    ///
    /// Missing variables are treated as unspecified. Malformed graph-source values
    /// are reported as configuration errors.
    fn read() -> Result<Self, String> {
        Ok(Self {
            graph_source: std::env::var("ORION_GRAPH")
                .ok()
                .map(|s| parse_graph_source(&s))
                .transpose()?,
            staged_file: std::env::var_os("ORION_STAGED_FILE").map(PathBuf::from),
            pa_root: std::env::var_os("PA_ROOT").map(PathBuf::from),
            legacy_local_pct: std::env::var("ORION_LOCAL_PCT")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|v| (1..=100).contains(v)),
            profiling: std::env::var_os("ORION_PROFILE_MARKER").is_some(),
        })
    }
}

/// Parses a legacy graph-source spelling accepted outside Clap's `ValueEnum`.
fn parse_graph_source(s: &str) -> Result<GraphSource, String> {
    match s.to_ascii_lowercase().as_str() {
        "rust" => Ok(GraphSource::Rust),
        "pa" | "parlayann" => Ok(GraphSource::Parlayann),
        _ => Err(format!(
            "unknown graph source {s:?}; expected rust or parlayann"
        )),
    }
}

impl Args {
    /// Resolves this CLI input into a complete runtime configuration.
    ///
    /// Resolution merges, in precedence order, explicit CLI values, legacy environment
    /// settings, dataset-specific YAML overrides, and global defaults.
    ///
    /// This method does not load vector payloads or construct/import graph data.
    ///
    /// # Errors
    ///
    /// Returns an error if the custom configuration file cannot be read or parsed,
    /// a referenced dataset preset does not exist, required settings are missing,
    /// or any resolved option combination is invalid.
    pub fn resolve(&self) -> Result<ResolvedRunConfig, String> {
        let custom_root: Option<RawRoot> = self
            .config
            .as_ref()
            .map(|path| {
                let text = std::fs::read_to_string(path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                serde_yaml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
            })
            .transpose()?;
        self.resolve_with(
            custom_root.as_ref().unwrap_or_else(|| load_root()),
            &LegacyEnvironment::read()?,
        )
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
    fn resolve_with(
        &self,
        root: &RawRoot,
        env: &LegacyEnvironment,
    ) -> Result<ResolvedRunConfig, String> {
        // Preserve the no-argument SIFT shortcut, but never assign it to custom paths.
        let dataset = self.dataset.clone().or_else(|| {
            if self.base.is_none() && self.query.is_none() && self.groundtruth.is_none() {
                Some("sift".into())
            } else {
                None
            }
        });
        let preset = dataset
            .as_ref()
            .map(|name| {
                root.datasets.get(name).ok_or_else(|| {
                    format!(
                        "unknown dataset shortcut {name:?}; add a YAML entry or omit the name and supply paths"
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
        let query = required_path(&self.query, paths.map(|p| &p.query), "query")?;
        let groundtruth = required_path(
            &self.groundtruth,
            paths.map(|p| &p.groundtruth),
            "groundtruth",
        )?;

        let base_format = self
            .base_format
            .map(Ok)
            .unwrap_or_else(|| VectorFormat::infer(&base))?;
        let query_format = self
            .query_format
            .map(Ok)
            .unwrap_or_else(|| VectorFormat::infer(&query))?;
        if self
            .memory_budget_gib
            .is_some_and(|x| !x.is_finite() || x <= 0.)
        {
            return Err("--memory-budget-gib must be finite and positive".into());
        }

        let metric = self
            .metric
            .or_else(|| preset.and_then(|p| p.metric))
            .ok_or(
                "--metric is required; do not infer a dataset objective from its admission kernel",
            )?;
        let dimension = match self.dimension.or_else(|| preset.map(|p| p.dimension)) {
            Some(d) => d,
            None => super::data::inspect_vector_file(&base, base_format)?.dimension,
        };
        crate::with_supported_dimension!(dimension, |D| Ok::<_, String>(D))?;

        let defaults = &root.defaults.orion;
        let dataset_overrides = preset.and_then(|p| p.orion.as_ref());
        let vector_storage = match self.vector_storage {
            Some(storage) => storage,
            None => match dataset_overrides.and_then(|o| o.vector_storage.as_deref()) {
                None | Some("f32") => VectorStorageKind::F32,
                Some("u8") => VectorStorageKind::U8,
                Some(other) => {
                    return Err(format!(
                        "unknown vector_storage {other:?}; expected f32 or u8"
                    ))
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
        if !(1..=64).contains(&window_size) || max_points == 0 {
            return Err("window-size must be 1..=64 and max-points positive".into());
        }

        let (graph_source, staged_file, staged_file_explicit) = self.resolve_graph_source(
            preset.and_then(|dataset| dataset.base_graph.as_ref()),
            env,
            max_extra,
        )?;

        // Native u8 storage is currently a deliberately narrow execution path.
        // It exists for SIFT-style 128-dimensional L2 data and reuses an imported
        // ParlayANN graph because the native Rust graph builder still operates on f32.
        if vector_storage == VectorStorageKind::U8 {
            if dimension != 128 || metric != SearchMetric::L2 {
                return Err(
                    "native u8 storage currently requires 128-dimensional L2 vectors".into(),
                );
            }
            if !matches!(base_format, VectorFormat::Bvecs | VectorFormat::U8bin)
                || !matches!(query_format, VectorFormat::Bvecs | VectorFormat::U8bin)
            {
                return Err(
                    "native u8 requires bvecs/u8bin base and queries; it does not quantize floats"
                        .into(),
                );
            }
            if cascade.prefilter != PrefilterChoice::None
                || cascade.admission != AdmissionChoice::L2U8
                || cascade.rerank != RerankChoice::None
            {
                return Err(
                    "native u8 requires --prefilter none --admission l2-u8 --rerank none".into(),
                );
            }
            if graph_source != GraphSource::Parlayann {
                return Err(
                    "native u8 uses a ParlayANN STAG export/cache; the Rust builder remains f32"
                        .into(),
                );
            }
        }

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

        let mut run = ResolvedRunConfig {
            dataset,
            base,
            query,
            groundtruth,
            base_format,
            query_format,
            vector_storage,
            memory_budget_gib: self.memory_budget_gib,
            prepare_only: self.prepare_only,
            dimension,
            metric,
            cascade,
            alpha,
            graph_degree,
            build_l,
            max_extra,
            window_size,
            graph_source,
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
            legacy_dataset: None,
            legacy_local_pct: env.legacy_local_pct.unwrap_or(60),
        };

        // Derive cache identities only after every effective setting is resolved.
        run.legacy_dataset = run.matching_legacy_preset();
        run.cache_namespace = run.compute_cache_namespace();

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
    ) -> Result<Cascade, String> {
        let mut cascade = Cascade::default_for_specified_dimension_and_metric(dimension, metric);
        if let Some(o) = dataset_overrides {
            if let Some(v) = &o.prefilter {
                cascade.prefilter = v.parse()?;
            }
            if let Some(v) = &o.admission {
                cascade.admission = v.parse()?;
            }
            if let Some(v) = &o.rerank {
                cascade.rerank = v.parse()?;
            }
        }

        // Explicit axes win over both the selector and the dataset preset.
        if let Some(v) = self.prefilter {
            cascade.prefilter = v;
        }
        if let Some(v) = self.admission {
            cascade.admission = v;
        }
        if let Some(v) = self.rerank {
            cascade.rerank = v;
        }

        cascade.validate_cascade_options(metric)?;

        Ok(cascade)
    }

    /// Resolves the graph implementation and optional staged graph path.
    ///
    /// Graph-source precedence is:
    ///
    /// 1. explicit CLI source or staged-file selection,
    /// 2. legacy environment settings,
    /// 3. dataset preset,
    /// 4. native Rust builder.
    ///
    /// The staged file is resolved but not opened here because an existing cache may
    /// remain valid even when its original staged export is no longer available.
    fn resolve_graph_source(
        &self,
        preset_graph: Option<&RawBaseGraph>,
        env: &LegacyEnvironment,
        max_extra: usize,
    ) -> Result<(GraphSource, Option<PathBuf>, bool), String> {
        let explicit_source = self
            .graph_source
            .or_else(|| self.staged_file.as_ref().map(|_| GraphSource::Parlayann))
            .or(env.graph_source)
            .or_else(|| env.staged_file.as_ref().map(|_| GraphSource::Parlayann));
        let graph_source = explicit_source.unwrap_or(
            preset_graph
                .and_then(|b| b.source.as_deref())
                .map(parse_graph_source)
                .transpose()?
                .unwrap_or(GraphSource::Rust),
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

        if graph_source == GraphSource::Rust && self.staged_file.is_some() {
            return Err("--staged-file conflicts with --graph-source rust".into());
        }
        Ok((graph_source, staged_file, staged_file_explicit))
    }
}

impl ResolvedRunConfig {
    /// Returns the bundled legacy preset whose historical cache identity exactly
    /// matches this resolved configuration.
    ///
    /// A matching dataset name alone is insufficient: the base path, storage format,
    /// metric, dimension, and graph-construction parameters must also match the
    /// bundled preset.
    ///
    /// This prevents custom configurations from accidentally reusing legacy caches.
    fn matching_legacy_preset(&self) -> Option<String> {
        // A custom YAML entry called "sift" is not evidence of the old cache's identity.
        let bundled: RawRoot =
            serde_yaml::from_str(include_str!("../../configs/sweep.yaml")).ok()?;
        if self.base_format != VectorFormat::Fvecs || self.vector_storage != VectorStorageKind::F32
        {
            return None;
        }
        let name = self.dataset.as_ref()?;
        let preset = bundled.datasets.get(name)?;
        let paths = preset.paths.as_ref()?;
        let ov = preset.orion.as_ref();
        let defaults = &bundled.defaults.orion;
        let same = |a: &std::path::Path, b: &std::path::Path| {
            a == b || matches!((a.canonicalize(), b.canonicalize()), (Ok(a), Ok(b)) if a == b)
        };
        if !same(&self.base, std::path::Path::new(&paths.base))
            || self.dimension != preset.dimension
            || Some(self.metric) != preset.metric
            || self.alpha != ov.and_then(|o| o.alpha).unwrap_or(defaults.alpha)
            || self.graph_degree
                != ov
                    .and_then(|o| o.graph_degree)
                    .unwrap_or(defaults.graph_degree)
            || self.build_l
                != ov
                    .and_then(|o| o.build_search_list_size)
                    .unwrap_or(defaults.build_search_list_size)
        {
            return None;
        }
        Some(name.clone())
    }

    /// Computes the stable namespace used for graph-cache files.
    ///
    /// The identity includes the base-file identity and every resolved setting that
    /// changes the stored graph. Query paths and search-only sweep settings are
    /// intentionally excluded.
    ///
    /// Existing `v3_*` cache identities depend on the serialized tuple layout below.
    /// Reordering fields or changing their encoding is therefore a cache-format change.
    ///
    /// Native `u8` storage additionally uses a dedicated namespace discriminator to
    /// prevent raw-byte graphs from colliding with historical `f32` cache entries.
    fn compute_cache_namespace(&self) -> String {
        fn base_file_identity(path: &std::path::Path) -> (PathBuf, Option<u64>, Option<u128>) {
            let absolute = std::fs::canonicalize(path).unwrap_or_else(|_| {
                if path.is_absolute() {
                    path.to_owned()
                } else {
                    std::env::current_dir()
                        .expect("working directory")
                        .join(path)
                }
            });
            let meta = std::fs::metadata(path).ok();
            let modified = meta
                .as_ref()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos());
            (absolute, meta.map(|m| m.len()), modified)
        }

        // Keep tuple order, field encoding, and the v3 prefix stable: these bytes
        // determine existing cache filenames. Query inputs intentionally do not
        // participate because they do not change the stored graph.
        let descriptor = serde_json::to_vec(&(
            base_file_identity(&self.base),
            self.dimension,
            self.metric,
            self.alpha,
            self.graph_degree,
            self.build_l,
            self.max_extra,
            self.graph_source,
            if self.graph_source == GraphSource::Parlayann {
                Some(self.legacy_local_pct)
            } else {
                None
            },
        ))
        .expect("serialize cache identity");
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        descriptor.hash(&mut hash);
        if self.base_format != VectorFormat::Fvecs {
            format!("{:?}", self.base_format).hash(&mut hash);
        }
        // Preserve all historical f32 identities; native storage gets its own
        // namespace so a stale quantized sidecar cannot be mistaken for raw bytes.
        if self.vector_storage == VectorStorageKind::U8 {
            "native-u8-v1".hash(&mut hash);
        }
        format!("v3_{:016x}", hash.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn root() -> RawRoot {
        serde_yaml::from_str(include_str!("../../configs/sweep.yaml")).unwrap()
    }
    fn resolve(args: &[&str]) -> Result<ResolvedRunConfig, String> {
        Args::try_parse_from(args)
            .map_err(|e| e.to_string())?
            .resolve_with(&root(), &LegacyEnvironment::default())
    }

    #[test]
    fn large_sift_presets_select_native_bytes_and_exact_prefixes() {
        for (name, count, groundtruth) in [
            ("sift10m", 10_000_000, "idx_10M.ivecs"),
            ("sift100m", 100_000_000, "idx_100M.ivecs"),
            ("sift1b", 1_000_000_000, "idx_1000M.ivecs"),
        ] {
            let config = resolve(&["orion", name]).unwrap();
            assert_eq!(config.vector_storage, VectorStorageKind::U8);
            assert_eq!(config.max_points, count);
            assert_eq!(config.base_format, VectorFormat::Bvecs);
            assert_eq!(config.query_format, VectorFormat::Bvecs);
            assert!(config.groundtruth.ends_with(groundtruth));
            assert_eq!(config.cascade.prefilter, PrefilterChoice::None);
            assert_eq!(config.cascade.admission, AdmissionChoice::L2U8);
            assert_eq!(config.cascade.rerank, RerankChoice::None);
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
        let float = resolve(&["orion", "sift10m", "--vector-storage", "f32"]).unwrap();
        assert_ne!(native.cache_namespace, float.cache_namespace);
        assert!(resolve(&["orion", "sift10m", "--rerank", "f32"]).is_err());
        assert!(resolve(&["orion", "sift10m", "--admission", "l2-u16"]).is_err());
        assert!(resolve(&["orion", "sift10m", "--prefilter", "jl"]).is_err());
        assert!(resolve(&["orion", "sift10m", "--query", "query.fvecs"]).is_err());
        assert!(resolve(&["orion", "sift10m", "--base", "base.fbin"]).is_err());
        assert!(resolve(&["orion", "sift10m", "--graph-source", "rust"]).is_err());
        assert!(resolve(&["orion", "sift10m", "--dimension", "32"]).is_err());
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
                1.05,
                100,
                200,
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
                c.cascade,
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
        assert!(c.dataset.is_none());
        assert_eq!(c.cascade.admission, AdmissionChoice::L2Kt);
        assert_eq!(c.sweep.calibration_l, 200);
        assert_eq!(c.graph_source, GraphSource::Rust);
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
        assert_eq!(b.cascade.prefilter, PrefilterChoice::Jl);
        assert_eq!(b.cascade.admission, a.cascade.admission);
        assert_ne!(a.cache_namespace, b.cache_namespace);
        let c = resolve(&["orion", "sift", "--graph-source", "rust", "--k", "100"]).unwrap();
        assert_eq!(a.cache_namespace, c.cache_namespace);
    }

    #[test]
    fn invalid_inputs_fail_before_execution() {
        for args in [
            vec!["orion", "missing"],
            vec!["orion", "sift", "--dimension", "25"],
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
            c.cascade,
            Cascade::default_for_specified_dimension_and_metric(960, SearchMetric::L2)
        );
    }

    #[test]
    fn cli_graph_source_overrides_legacy_environment() {
        let args = Args::try_parse_from(["orion", "sift", "--graph-source", "rust"]).unwrap();
        let env = LegacyEnvironment {
            graph_source: Some(GraphSource::Parlayann),
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
        assert_eq!(default.legacy_dataset.as_deref(), Some("sift"));
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
        assert!(config.legacy_dataset.is_none());
    }

    #[test]
    fn custom_input_and_config_cannot_borrow_a_named_legacy_cache() {
        let custom = resolve(&["orion", "sift", "--base", "different.fvecs"]).unwrap();
        assert!(custom.legacy_dataset.is_none());
        let changed_alpha = resolve(&["orion", "sift", "--alpha", "1.151"]).unwrap();
        assert!(changed_alpha.legacy_dataset.is_none());
        let mut custom_root = root();
        custom_root
            .datasets
            .get_mut("sift")
            .unwrap()
            .paths
            .as_mut()
            .unwrap()
            .base = "different.fvecs".into();
        let args = Args::try_parse_from(["orion", "sift"]).unwrap();
        let config = args
            .resolve_with(&custom_root, &LegacyEnvironment::default())
            .unwrap();
        assert!(config.legacy_dataset.is_none());
    }
}
