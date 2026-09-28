/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resolve presets and CLI overrides before loading vectors or running search.
use crate::config::{
    load_root, try_resolve_placeholders, RawRoot, ResolvedSweep, SweepOverrides,
};
use crate::cascade::{AdmissionChoice, Cascade, PrefilterChoice, RerankChoice, SearchMetric};
use clap::Parser;
use super::data::VectorFormat;
use serde::Serialize;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum GraphSource {
    Rust,
    #[value(alias = "pa")]
    Parlayann,
}

#[derive(Debug, Parser)]
#[command(name = "orion", about = "Configurable Orion QPS-recall sweep")]
pub struct Args {
    /// Optional dataset shortcut from sweep.yaml; defaults to sift with no paths.
    pub dataset: Option<String>,
    /// Legacy second positional argument.
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
    /// Print input and resident-memory lower bounds without loading base or graph.
    #[arg(long)]
    pub preflight: bool,
    /// Reject when known resident components exceed this GiB budget. Not a peak guarantee.
    #[arg(long)]
    pub memory_budget_gib: Option<f64>,
    /// Build/import and save graph cache, then exit before calibration/search.
    #[arg(long)]
    pub prepare_only: bool,
    /// Physical dimension in each fvecs record. Inferred from base header if omitted.
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

#[derive(Debug, Serialize)]
pub struct ResolvedRunConfig {
    pub dataset: Option<String>,
    pub base: PathBuf,
    pub query: PathBuf,
    pub groundtruth: PathBuf,
    pub base_format: VectorFormat,
    pub query_format: VectorFormat,
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
    pub cache_namespace: String,
    pub cache_dir: PathBuf,
    #[serde(skip)]
    pub staged_file_explicit: bool,
    #[serde(skip)]
    pub legacy_dataset: Option<String>,
    #[serde(skip)]
    pub legacy_local_pct: usize,
}

#[derive(Default)]
struct Environment {
    graph_source: Option<GraphSource>,
    staged_file: Option<PathBuf>,
    pa_root: Option<PathBuf>,
    legacy_local_pct: Option<usize>,
    profiling: bool,
}

impl Environment {
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
    /// Only YAML file is valid under current version. The input standard should be a superset of
    /// the example config file listed in `configs/sweep.yaml` which means that every element showed
    /// in the example file is needed. Append a new configuration for frequently used dataset in
    /// `configs/sweep.yaml` is a recommended option.
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
            &Environment::read()?,
        )
    }

    fn resolve_with(&self, root: &RawRoot, env: &Environment) -> Result<ResolvedRunConfig, String> {
        // May change the default dataset chosen strategy in the future.
        let dataset = self.dataset.clone().or_else(|| {
            if self.base.is_none() && self.query.is_none() && self.groundtruth.is_none() {
                Some("sift".into())
            } else {
                None
            }
        });
        let preset = dataset.as_ref().map(|name| root.datasets.get(name)
            .ok_or_else(|| format!("unknown dataset shortcut {name:?}; add a YAML entry or omit the name and supply paths")))
            .transpose()?;
        let paths = preset.and_then(|p| p.paths.as_ref());

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

        let base_format = self.base_format.map(Ok).unwrap_or_else(|| VectorFormat::infer(&base))?;
        let query_format = self.query_format.map(Ok).unwrap_or_else(|| VectorFormat::infer(&query))?;
        if self.memory_budget_gib.is_some_and(|x| !x.is_finite() || x <= 0.) {
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
            None => super::data::inspect(&base, base_format)?.dimension,
        };
        crate::with_supported_dimension!(dimension, |D| Ok::<_, String>(D))?;

        let defaults = &root.defaults.orion;
        let ov = preset.and_then(|p| p.orion.as_ref());
        let mut cascade = Cascade::default_for_specified_dimension_and_metric(dimension, metric);
        if let Some(o) = ov {
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

        let alpha = self
            .alpha
            .or_else(|| ov.and_then(|o| o.alpha))
            .unwrap_or(defaults.alpha);
        let graph_degree = self
            .graph_degree
            .or_else(|| ov.and_then(|o| o.graph_degree))
            .unwrap_or(defaults.graph_degree);
        let build_l = self
            .build_l
            .or_else(|| ov.and_then(|o| o.build_search_list_size))
            .unwrap_or(defaults.build_search_list_size);
        let max_extra = self
            .max_extra
            .or_else(|| ov.and_then(|o| o.max_extra))
            .unwrap_or(defaults.max_extra);
        let window_size = self
            .window_size
            .or_else(|| ov.and_then(|o| o.window_size))
            .unwrap_or(defaults.window_size);
        let max_points = self
            .max_points
            .or(self.legacy_max_points)
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

        let bg = if preset.is_some() {
            preset.and_then(|p| p.base_graph.as_ref())
        } else {
            None
        };

        let explicit_source = self
            .graph_source
            .or_else(|| self.staged_file.as_ref().map(|_| GraphSource::Parlayann))
            .or(env.graph_source)
            .or_else(|| env.staged_file.as_ref().map(|_| GraphSource::Parlayann));
        let graph_source = explicit_source.unwrap_or(
            bg.and_then(|b| b.source.as_deref())
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
                bg.and_then(|b| b.staged_file.as_ref())
                    .and_then(|s| {
                        try_resolve_placeholders(
                            &s
                                .replace("${max_extra}", &max_extra.to_string())
                                .replace("${PA_ROOT}", &pa_root.to_string_lossy()),
                        )
                    })
                    .map(PathBuf::from)
            });

        // An export is an import dependency, not a cache-loading dependency.
        // Validate it only after the runtime cache lookup misses.

        if graph_source == GraphSource::Rust && self.staged_file.is_some() {
            return Err("--staged-file conflicts with --graph-source rust".into());
        }
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
            base_format, query_format,
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
        run.legacy_dataset = run.matching_legacy_preset();
        run.cache_namespace = run.cache_identity();

        Ok(run)
    }
}

pub fn read_dimension(path: &std::path::Path) -> Result<usize, String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut bytes = [0; 4];
    file.read_exact(&mut bytes)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(u32::from_le_bytes(bytes) as usize)
}

impl ResolvedRunConfig {
    fn matching_legacy_preset(&self) -> Option<String> {
        // A custom YAML entry called "sift" is not evidence of the old cache's identity.
        let bundled: RawRoot = serde_yaml::from_str(include_str!("../../configs/sweep.yaml")).ok()?;
        if self.base_format != VectorFormat::Fvecs { return None; }
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

    fn cache_identity(&self) -> String {
        fn identity(path: &std::path::Path) -> (PathBuf, Option<u64>, Option<u128>) {
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
        // Versioned namespace: never interpret a legacy name-only cache as custom data.
        let descriptor = serde_json::to_vec(&(
            identity(&self.base),
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
            .resolve_with(&root(), &Environment::default())
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
            .resolve_with(&root, &Environment::default())
            .unwrap();
        assert_eq!(
            c.cascade,
            Cascade::default_for_specified_dimension_and_metric(960, SearchMetric::L2)
        );
    }

    #[test]
    fn cli_graph_source_overrides_legacy_environment() {
        let args = Args::try_parse_from(["orion", "sift", "--graph-source", "rust"]).unwrap();
        let env = Environment {
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
        let original = c.cache_identity();
        c.alpha += 0.001;
        assert_ne!(original, c.cache_identity());
        c.metric = SearchMetric::InnerProduct;
        let ip = c.cache_identity();
        c.metric = SearchMetric::Cosine;
        assert_ne!(ip, c.cache_identity());
    }

    #[test]
    fn pa_root_is_optional_and_does_not_identify_the_cache() {
        let args = Args::try_parse_from(["orion", "sift", "--graph-source", "parlayann"]).unwrap();
        let default = args.resolve_with(&root(), &Environment::default()).unwrap();
        assert_eq!(default.graph_source, GraphSource::Parlayann);
        assert!(default
            .staged_file
            .as_ref()
            .unwrap()
            .starts_with("../ParlayANN"));
        let elsewhere = args
            .resolve_with(
                &root(),
                &Environment {
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
        let config = args.resolve_with(&root(), &Environment::default()).unwrap();
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
            .resolve_with(&custom_root, &Environment::default())
            .unwrap();
        assert!(config.legacy_dataset.is_none());
    }
}
