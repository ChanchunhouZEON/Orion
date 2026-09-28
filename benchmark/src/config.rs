/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Central config loader for `benchmark/configs/sweep.yaml`.
//!
//! Every benchmark entry point (`run_qps_recall_sweep`,
//! `run_orion_parlayann_sweep`, and the secondary profiling utilities)
//! resolves its per-dataset parameters through [`load_dataset_config`].
//! The YAML file is the single source of truth for:
//!
//!   * data file paths (`paths.{base, query, groundtruth}`)
//!   * build parameters (`orion.{alpha, graph_degree, build_l, max_extra,
//!     window_size}`)
//!   * dataset objective (`metric`: l2, inner-product, or cosine)
//!   * base-graph source (`base_graph.{source, staged_file}` —
//!     `rust` = build in-process; `parlayann` = import a `.staged v2`
//!     export)
//!
//! Placeholders in path strings (`${PA_ROOT}`, `${max_extra}`) are
//! resolved at load time. `${PA_ROOT}` must be set via env (no
//! repo-baked default — keeps user paths out of committed config);
//! `${max_extra}` resolves from the dataset's `orion.max_extra`.
//!
//! Many fields are deserialised by serde from YAML but consumed only
//! in subsystems that aren't reached from every benchmark binary —
//! suppress the `dead_code` warning at the module level rather than
//! tagging every struct field.

#![allow(dead_code)]

use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

// ── User-facing enums ──────────────────────────────────────────────────────

/// Where the base graph comes from.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum BaseGraphSource {
    /// Build in-process via `build_diskann_index`.
    Rust,
    /// Import a ParlayANN `.staged v2` export.
    Parlayann,
}

/// Fully resolved config for one dataset.
#[derive(Clone, Debug)]
pub struct DatasetConfig {
    pub name: String,
    pub dimension: usize,
    pub base_path: PathBuf,
    pub query_path: PathBuf,
    pub groundtruth_path: PathBuf,
    pub diskann: DiskANNConfig,
    pub orion: OrionConfig,
    pub base_graph: BaseGraphConfig,
    pub sweep: SweepConfig,
}

#[derive(Clone, Debug)]
pub struct OrionConfig {
    pub alpha: f32,
    pub graph_degree: u32,
    pub build_search_list_size: usize,
    pub max_extra: usize,
    pub window_size: usize,
    /// Cascade triple loaded directly from `sweep.yaml`. The unified
    /// search pipeline reads these to build the prefilter / admission
    /// / rerank stages. Mirrors the `Cascade::default_for_dataset`
    /// mapping in `orion.rs`.
    pub prefilter: crate::cascade::PrefilterChoice,
    pub admission: crate::cascade::AdmissionChoice,
    pub rerank: crate::cascade::RerankChoice,
}

#[derive(Clone, Debug)]
pub struct BaseGraphConfig {
    pub source: BaseGraphSource,
    /// Only populated when `source == Parlayann`.
    pub staged_file: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SweepConfig {
    pub k: usize,
    pub threads: usize,
    pub trials: usize,
    pub calibration: CalibrationSettings,
    #[serde(default)]
    pub profiles: HashMap<String, ProfileDefaults>,
    pub schedules_by_k: HashMap<usize, HashMap<String, Vec<usize>>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CalibrationSettings {
    pub samples: usize,
    pub base_l: usize,
    pub diagnostic_l: usize,
}

impl CalibrationSettings {
    pub fn search_list_size(&self, k: usize) -> usize {
        orion::calibration_search_list_size!(self.base_l, k)
    }

    pub fn diagnostic_search_list_size(&self, k: usize) -> usize {
        orion::calibration_search_list_size!(self.diagnostic_l, k)
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ProfileDefaults {
    pub threads: Option<usize>,
    pub trials: Option<usize>,
}

#[derive(Clone, Debug, Default, clap::Args)]
pub struct SweepOverrides {
    #[arg(long, alias = "ls", value_delimiter = ',')]
    pub search_list_sizes: Option<Vec<usize>>,
    #[arg(long)]
    pub threads: Option<usize>,
    #[arg(long)]
    pub trials: Option<usize>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ResolvedSweep {
    pub k: usize,
    pub profile: String,
    pub search_list_sizes: Vec<usize>,
    pub threads: usize,
    pub trials: usize,
    pub calibration_samples: usize,
    pub calibration_l: usize,
}

impl SweepConfig {
    pub fn resolve(
        &self,
        profile: &str,
        k: usize,
        overrides: &SweepOverrides,
    ) -> Result<ResolvedSweep, String> {
        let search_list_sizes = overrides.search_list_sizes.clone().or_else(|| {
            self.schedules_by_k.get(&k).and_then(|row| row.get(profile)).cloned()
        }).ok_or_else(|| format!("no sweep.schedules_by_k entry for k={k}, profile={profile}; add one or pass --search-list-sizes"))?;
        crate::utils::validate_search_list_sizes(k, &search_list_sizes)?;
        let defaults = self.profiles.get(profile);
        let threads = overrides
            .threads
            .or_else(|| defaults.and_then(|p| p.threads))
            .unwrap_or(self.threads);
        let trials = overrides
            .trials
            .or_else(|| defaults.and_then(|p| p.trials))
            .unwrap_or(self.trials);
        if threads == 0
            || trials == 0
            || self.calibration.samples == 0
            || self.calibration.base_l == 0
            || self.calibration.diagnostic_l == 0
        {
            return Err(
                "threads, trials, calibration samples, base_l and diagnostic_l must be positive"
                    .into(),
            );
        }
        Ok(ResolvedSweep {
            k,
            profile: profile.into(),
            search_list_sizes,
            threads,
            trials,
            calibration_samples: self.calibration.samples,
            calibration_l: self.calibration.search_list_size(k),
        })
    }
}

/// DiskANN baseline config — kept flat since the L2 baseline doesn't
/// vary per-dataset beyond defaults today.
#[derive(Clone, Debug)]
pub struct DiskANNConfig {
    pub alpha: f32,
    pub graph_degree: u32,
    pub build_search_list_size: usize,
}

// ── Raw YAML ↔ struct layer ────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct RawRoot {
    pub defaults: RawDefaults,
    pub datasets: HashMap<String, RawDataset>,
}

#[derive(Deserialize)]
pub struct RawDefaults {
    pub diskann: RawDiskANNDefaults,
    pub orion: RawOrionDefaults,
    pub sweep: SweepConfig,
}

#[derive(Deserialize)]
pub struct RawDiskANNDefaults {
    pub alpha: f32,
    pub graph_degree: u32,
    pub build_search_list_size: usize,
}

#[derive(Deserialize)]
pub struct RawOrionDefaults {
    pub alpha: f32,
    pub graph_degree: u32,
    pub build_search_list_size: usize,
    pub max_extra: usize,
    pub window_size: usize,
    pub prefilter: String,
    pub admission: String,
    pub rerank: String,
}

#[derive(Deserialize, Default, Debug)]
pub struct RawDataset {
    pub dimension: usize,
    pub metric: Option<crate::cascade::SearchMetric>,
    pub paths: Option<RawPaths>,
    pub base_graph: Option<RawBaseGraph>,
    pub diskann: Option<RawDiskANNOverride>,
    pub orion: Option<RawOrionOverride>,
}

#[derive(Deserialize, Default, Debug)]
pub struct RawDiskANNOverride {
    pub alpha: Option<f32>,
    pub graph_degree: Option<u32>,
    pub build_search_list_size: Option<usize>,
}

#[derive(Deserialize, Default, Debug)]
pub struct RawPaths {
    pub base: String,
    pub query: String,
    pub groundtruth: String,
}

#[derive(Deserialize, Default, Debug)]
pub struct RawBaseGraph {
    pub source: Option<String>,
    pub staged_file: Option<String>,
}

#[derive(Deserialize, Default, Debug)]
pub struct RawOrionOverride {
    pub alpha: Option<f32>,
    pub graph_degree: Option<u32>,
    pub build_search_list_size: Option<usize>,
    pub max_extra: Option<usize>,
    pub window_size: Option<usize>,
    pub prefilter: Option<String>,
    pub admission: Option<String>,
    pub rerank: Option<String>,
}

// ── Public API ─────────────────────────────────────────────────────────────

const DEFAULT_CONFIG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/configs/sweep.yaml");

pub fn load_root() -> &'static RawRoot {
    static ROOT: std::sync::OnceLock<RawRoot> = std::sync::OnceLock::new();
    ROOT.get_or_init(|| {
        let path =
            std::env::var("ORION_SWEEP_CONFIG").unwrap_or_else(|_| DEFAULT_CONFIG_PATH.to_string());
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("Cannot read {path}: {e}"));
        serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("Invalid YAML in {path}: {e}"))
    })
}

pub fn load_sweep_config() -> &'static SweepConfig {
    &load_root().defaults.sweep
}

/// Load config for a single dataset by name. Panics on missing dataset
/// or malformed YAML — the file is a build-time contract, not user
/// input, so early failure is the right response.
pub fn load_dataset_config(name: &str) -> DatasetConfig {
    resolve_dataset(load_root(), name)
}

/// Lookup dataset by the dimension-to-name mapping the harness used
/// historically. Useful for `main.rs` sites that only have `dim` at hand.
pub fn load_dataset_config_by_dim(dimension: usize) -> DatasetConfig {
    let name = match dimension {
        32 => "glove25",
        100 => "glove100",
        128 => "sift",
        768 => "msmarco_bert_1M",
        960 => "gist",
        1536 => "wiki_ada_1M",
        _ => panic!("Unsupported dimension: {dimension} (no dataset entry in sweep.yaml)"),
    };
    load_dataset_config(name)
}

/// Global DiskANN defaults — only used when no dataset is in scope
/// (e.g. utility scripts). Per-dataset DiskANN config travels on
/// `DatasetConfig::diskann` with field-level fallback to orion → defaults.
pub fn load_diskann_defaults() -> DiskANNConfig {
    let path =
        std::env::var("ORION_SWEEP_CONFIG").unwrap_or_else(|_| DEFAULT_CONFIG_PATH.to_string());
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("Cannot read {path}: {e}"));
    let root: RawRoot =
        serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("Invalid YAML in {path}: {e}"));
    DiskANNConfig {
        alpha: root.defaults.diskann.alpha,
        graph_degree: root.defaults.diskann.graph_degree,
        build_search_list_size: root.defaults.diskann.build_search_list_size,
    }
}

// ── Internals ──────────────────────────────────────────────────────────────

fn resolve_dataset(root: &RawRoot, name: &str) -> DatasetConfig {
    let ds = root
        .datasets
        .get(name)
        .unwrap_or_else(|| panic!("Dataset '{name}' not in sweep.yaml"));

    let paths = ds
        .paths
        .as_ref()
        .unwrap_or_else(|| panic!("Dataset '{name}' missing `paths` block"));

    let orion_d = &root.defaults.orion;
    let ov = ds.orion.as_ref();
    let orion = OrionConfig {
        alpha: ov.and_then(|o| o.alpha).unwrap_or(orion_d.alpha),
        graph_degree: ov
            .and_then(|o| o.graph_degree)
            .unwrap_or(orion_d.graph_degree),
        build_search_list_size: ov
            .and_then(|o| o.build_search_list_size)
            .unwrap_or(orion_d.build_search_list_size),
        max_extra: ov.and_then(|o| o.max_extra).unwrap_or(orion_d.max_extra),
        window_size: ov
            .and_then(|o| o.window_size)
            .unwrap_or(orion_d.window_size),
        prefilter: parse_cascade_str::<crate::cascade::PrefilterChoice>(
            ov.and_then(|o| o.prefilter.as_deref())
                .unwrap_or(orion_d.prefilter.as_str()),
            "prefilter",
        ),
        admission: parse_cascade_str::<crate::cascade::AdmissionChoice>(
            ov.and_then(|o| o.admission.as_deref())
                .unwrap_or(orion_d.admission.as_str()),
            "admission",
        ),
        rerank: parse_cascade_str::<crate::cascade::RerankChoice>(
            ov.and_then(|o| o.rerank.as_deref())
                .unwrap_or(orion_d.rerank.as_str()),
            "rerank",
        ),
    };

    // DiskANN baseline params — per-dataset `diskann:` override block
    // wins; otherwise fall back to `defaults.diskann` (sweep.yaml top
    // block). This keeps the Origin row in ablation panels honest:
    // it reports a vanilla DiskANN baseline at the *published*
    // defaults (α=2.0, R=64, L_build=100), not at a graph shape that
    // silently mirrors `orion.{R,L,α}` and would conflate "DiskANN
    // is slow" with "Vamana at orion's R is slow." The 3-engine
    // head-to-head's DiskANN row resolves the same way.
    let diskann_d = &root.defaults.diskann;
    let dov = ds.diskann.as_ref();
    let diskann = DiskANNConfig {
        alpha: dov.and_then(|o| o.alpha).unwrap_or(diskann_d.alpha),
        graph_degree: dov
            .and_then(|o| o.graph_degree)
            .unwrap_or(diskann_d.graph_degree),
        build_search_list_size: dov
            .and_then(|o| o.build_search_list_size)
            .unwrap_or(diskann_d.build_search_list_size),
    };

    let bg_raw = ds.base_graph.as_ref();
    let mut source = match bg_raw.and_then(|b| b.source.as_deref()).unwrap_or("rust") {
        "rust" | "Rust" => BaseGraphSource::Rust,
        "parlayann" | "Parlayann" | "PA" | "pa" => BaseGraphSource::Parlayann,
        other => panic!("Unknown base_graph.source: {other} (expected rust|parlayann)"),
    };
    // `staged_file` may reference `${max_extra}` (dataset-context) so
    // the path stays in sync with whatever `orion.max_extra` is set
    // at, plus env-var placeholders like `${PA_ROOT}`. Substitute
    // `${max_extra}` first, then try to resolve env vars — if any
    // env var is unset (typically `PA_ROOT` on a dev box without a
    // ParlayANN checkout), fall back to the Rust in-process build
    // path instead of panicking. Lets every benchmark profile run
    // out of the box without requiring PA_ROOT.
    let mut staged_file: Option<PathBuf> = None;
    if let Some(s) = bg_raw.and_then(|b| b.staged_file.clone()) {
        let max_extra_subbed = s.replace("${max_extra}", &orion.max_extra.to_string());
        match try_resolve_placeholders(&max_extra_subbed) {
            Some(resolved) => staged_file = Some(PathBuf::from(resolved)),
            None => {
                if matches!(source, BaseGraphSource::Parlayann) {
                    eprintln!(
                        "config: dataset '{name}' has base_graph.source=parlayann but a \
                         placeholder in staged_file '{s}' isn't resolvable (env var unset) \
                         — falling back to source=rust (build graph in-process)."
                    );
                    source = BaseGraphSource::Rust;
                }
                // staged_file stays None.
            }
        }
    }
    if matches!(source, BaseGraphSource::Parlayann) && staged_file.is_none() {
        panic!("Dataset '{name}' has base_graph.source=parlayann but no staged_file");
    }

    DatasetConfig {
        name: name.to_string(),
        dimension: ds.dimension,
        base_path: PathBuf::from(paths.base.clone()),
        query_path: PathBuf::from(paths.query.clone()),
        groundtruth_path: PathBuf::from(paths.groundtruth.clone()),
        diskann,
        orion,
        base_graph: BaseGraphConfig {
            source,
            staged_file,
        },
        sweep: root.defaults.sweep.clone(),
    }
}

/// Replace `${VAR}` occurrences with the corresponding env var, returning
/// `None` if any required var is unset. Lets callers (e.g.
/// `resolve_dataset`) fall back to a Rust in-process build path when
/// `PA_ROOT` isn't available, instead of forcing every profile invocation
/// to set it. Unterminated `${...}` is still a malformed-config panic.
pub fn try_resolve_placeholders(s: &str) -> Option<String> {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'$' && bytes[i + 1] == b'{' {
            let close = s[i + 2..]
                .find('}')
                .unwrap_or_else(|| panic!("unterminated ${{...}} in {s:?}"))
                + i
                + 2;
            let var = &s[i + 2..close];
            let val = match std::env::var(var) {
                Ok(v) => v,
                Err(_) => default_for(var)?,
            };
            out.push_str(&val);
            i = close + 1;
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    Some(out)
}

/// Hardcoded defaults for placeholder vars. Empty by design —
/// `PA_ROOT` (and any other path placeholder) must be supplied via
/// env so the repo doesn't ship user-specific paths.
fn default_for(_var: &str) -> Option<String> {
    None
}

/// Parse a cascade-choice string into the corresponding enum, with a
/// helpful error message keyed on which axis we're reading. Wraps the
/// per-enum `FromStr` impls so the YAML loader can fail fast with a
/// pointed message on a typo.
fn parse_cascade_str<T: std::str::FromStr<Err = String>>(s: &str, axis: &str) -> T {
    T::from_str(s).unwrap_or_else(|e| panic!("sweep.yaml orion.{axis} = {s:?} — {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sweep() -> SweepConfig {
        serde_yaml::from_str::<RawRoot>(include_str!("../configs/sweep.yaml"))
            .unwrap()
            .defaults
            .sweep
    }

    #[test]
    fn default_schedules_preserve_top10_and_support_top100() {
        let cfg = sweep();
        let overrides = SweepOverrides::default();
        let ten = cfg.resolve("sweep", 10, &overrides).unwrap();
        assert_eq!(
            ten.search_list_sizes,
            vec![
                16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 90, 100, 114,
                128, 144, 160, 180, 200, 224, 256, 288, 320, 384, 448, 512, 640, 768, 1024,
            ]
        );
        assert_eq!(
            (
                ten.threads,
                ten.trials,
                ten.calibration_samples,
                ten.calibration_l
            ),
            (8, 1, 200, 48)
        );
        let ablation = cfg.resolve("ablation", 10, &overrides).unwrap();
        assert_eq!(ablation.trials, 5);
        assert_eq!(ablation.search_list_sizes.len(), 14);
        let hundred = cfg.resolve("sweep", 100, &overrides).unwrap();
        assert_eq!(hundred.calibration_l, 200);
        assert_eq!(hundred.search_list_sizes[0], 100);
        for (&k, row) in &cfg.schedules_by_k {
            for profile in row.keys() {
                cfg.resolve(profile, k, &overrides).unwrap();
            }
        }
    }

    #[test]
    fn cli_overrides_profile_and_yaml_defaults() {
        let cfg = sweep();
        let overrides = SweepOverrides {
            search_list_sizes: Some(vec![64, 128]),
            threads: Some(2),
            trials: Some(3),
        };
        let resolved = cfg.resolve("ablation", 50, &overrides).unwrap();
        assert_eq!(resolved.search_list_sizes, vec![64, 128]);
        assert_eq!(
            (resolved.threads, resolved.trials, resolved.calibration_l),
            (2, 3, 100)
        );
        assert!(cfg
            .resolve("sweep", 50, &SweepOverrides::default())
            .is_err());
    }

    #[test]
    fn rejects_invalid_schedules_and_counts() {
        let cfg = sweep();
        for ls in [vec![], vec![16, 100], vec![100, 100], vec![200, 100]] {
            let overrides = SweepOverrides {
                search_list_sizes: Some(ls),
                ..Default::default()
            };
            assert!(cfg.resolve("sweep", 100, &overrides).is_err());
        }
        for overrides in [
            SweepOverrides {
                threads: Some(0),
                ..Default::default()
            },
            SweepOverrides {
                trials: Some(0),
                ..Default::default()
            },
        ] {
            assert!(cfg.resolve("sweep", 10, &overrides).is_err());
        }
        let overrides = SweepOverrides {
            search_list_sizes: Some(vec![16]),
            ..Default::default()
        };
        assert!(cfg.resolve("sweep", 0, &overrides).is_err());
    }

    #[test]
    fn calibration_values_come_from_yaml_settings() {
        let mut cfg = sweep();
        cfg.calibration.samples = 300;
        cfg.calibration.base_l = 256;
        cfg.calibration.diagnostic_l = 512;
        let run = cfg
            .resolve("sweep", 100, &SweepOverrides::default())
            .unwrap();
        assert_eq!((run.calibration_samples, run.calibration_l), (300, 256));
        assert_eq!(cfg.calibration.diagnostic_search_list_size(100), 512);
        cfg.calibration.samples = 0;
        assert!(cfg
            .resolve("sweep", 100, &SweepOverrides::default())
            .is_err());
    }

    #[test]
    fn placeholder_resolves_pa_root_env() {
        // SAFETY: env var mutation is process-global, but other tests
        // here don't read PA_ROOT, and serial-by-default `cargo test`
        // run order makes this safe. Set + verify + clean up.
        let prev = std::env::var("PA_ROOT").ok();
        // SAFETY: see comment above re. process-global env.
        unsafe {
            std::env::set_var("PA_ROOT", "/tmp/pa-test-root");
        }
        let resolved = try_resolve_placeholders("${PA_ROOT}/data/glove100/x.staged").unwrap();
        assert_eq!(resolved, "/tmp/pa-test-root/data/glove100/x.staged");
        match prev {
            Some(v) => unsafe {
                std::env::set_var("PA_ROOT", v);
            },
            None => unsafe {
                std::env::remove_var("PA_ROOT");
            },
        }
    }
}
