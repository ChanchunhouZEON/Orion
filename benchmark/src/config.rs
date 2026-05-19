/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Central config loader for `benchmark/configs/sweep.yaml`.
//!
//! Every benchmark entry point (`run_qps_recall_sweep`,
//! `run_staged_parlayann_sweep`, and the secondary profiling utilities)
//! resolves its per-dataset parameters through [`load_dataset_config`].
//! The YAML file is the single source of truth for:
//!
//!   * data file paths (`paths.{base, query, groundtruth}`)
//!   * build parameters (`staged.{alpha, graph_degree, build_l, max_extra,
//!     window_size}`)
//!   * search metric (`staged.metric` ∈ `l2` | `mips` | `mips-q`)
//!   * base-graph source (`base_graph.{source, staged_file}` —
//!     `rust` = build in-process; `parlayann` = import a `.staged v2`
//!     export)
//!
//! Placeholders in path strings (`${PA_ROOT}`, `${max_extra}`) are
//! resolved at load time. `${PA_ROOT}` must be set via env (no
//! repo-baked default — keeps user paths out of committed config);
//! `${max_extra}` resolves from the dataset's `staged.max_extra`.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

// ── User-facing enums ──────────────────────────────────────────────────────

/// Search metric selected per-dataset. Drives both the distance kernel
/// and (for MIPS-family) whether the base vectors are normalized at
/// ingest.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Metric {
    /// 2-phase u8 prefilter + f32 L2 rerank (production non-angular).
    L2,
    ///  PA-style i8 beam + f32 top-`k × 2` rerank.
    L2Q,
    /// Single-phase `-⟨q, v⟩` on unit-normalized data.
    Mips,
    /// PA-style i8 beam + f32 top-`k × 2` rerank (angular high-dim).
    MipsQ,
}

impl Metric {
    pub fn parse(s: &str) -> Self {
        match s {
            "l2" | "L2" => Metric::L2,
            "l2-q" | "l2_q" | "L2-Q" => Metric::L2Q,
            "mips" | "MIPS" => Metric::Mips,
            "mips-q" | "mips_q" | "MIPS-Q" => Metric::MipsQ,
            _ => panic!("Unknown metric: {s} (expected l2|l2-q|mips|mips-q)"),
        }
    }

    pub fn is_mips_family(self) -> bool {
        matches!(self, Metric::Mips | Metric::MipsQ)
    }
}

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
    pub staged: StagedConfig,
    pub base_graph: BaseGraphConfig,
    pub sweep: SweepConfig,
}

#[derive(Clone, Debug)]
pub struct StagedConfig {
    pub alpha: f32,
    pub graph_degree: u32,
    pub build_search_list_size: usize,
    pub max_extra: usize,
    pub window_size: usize,
    pub metric: Metric,
}

#[derive(Clone, Debug)]
pub struct BaseGraphConfig {
    pub source: BaseGraphSource,
    /// Only populated when `source == Parlayann`.
    pub staged_file: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct SweepConfig {
    pub search_list_sizes: Vec<usize>,
    pub threads: usize,
    pub trials: usize,
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
struct RawRoot {
    defaults: RawDefaults,
    datasets: HashMap<String, RawDataset>,
}

#[derive(Deserialize)]
struct RawDefaults {
    diskann: RawDiskANNDefaults,
    staged: RawStagedDefaults,
    sweep: RawSweepDefaults,
}

#[derive(Deserialize)]
struct RawDiskANNDefaults {
    alpha: f32,
    graph_degree: u32,
    build_search_list_size: usize,
}

#[derive(Deserialize)]
struct RawStagedDefaults {
    alpha: f32,
    graph_degree: u32,
    build_search_list_size: usize,
    max_extra: usize,
    window_size: usize,
    metric: String,
}

#[derive(Deserialize)]
struct RawSweepDefaults {
    search_list_sizes: Vec<usize>,
    threads: usize,
    trials: usize,
}

#[derive(Deserialize, Default)]
struct RawDataset {
    dimension: usize,
    paths: Option<RawPaths>,
    base_graph: Option<RawBaseGraph>,
    diskann: Option<RawDiskANNOverride>,
    staged: Option<RawStagedOverride>,
}

#[derive(Deserialize, Default)]
struct RawDiskANNOverride {
    alpha: Option<f32>,
    graph_degree: Option<u32>,
    build_search_list_size: Option<usize>,
}

#[derive(Deserialize, Default)]
struct RawPaths {
    base: String,
    query: String,
    groundtruth: String,
}

#[derive(Deserialize, Default)]
struct RawBaseGraph {
    source: Option<String>,
    staged_file: Option<String>,
}

#[derive(Deserialize, Default)]
struct RawStagedOverride {
    alpha: Option<f32>,
    graph_degree: Option<u32>,
    build_search_list_size: Option<usize>,
    max_extra: Option<usize>,
    window_size: Option<usize>,
    metric: Option<String>,
}

// ── Public API ─────────────────────────────────────────────────────────────

const DEFAULT_CONFIG_PATH: &str = "benchmark/configs/sweep.yaml";

/// Load config for a single dataset by name. Panics on missing dataset
/// or malformed YAML — the file is a build-time contract, not user
/// input, so early failure is the right response.
pub fn load_dataset_config(name: &str) -> DatasetConfig {
    let path =
        std::env::var("STAGED_SWEEP_CONFIG").unwrap_or_else(|_| DEFAULT_CONFIG_PATH.to_string());
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("Cannot read {path}: {e}"));
    let root: RawRoot =
        serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("Invalid YAML in {path}: {e}"));
    resolve_dataset(&root, name)
}

/// Lookup dataset by the dimension-to-name mapping the harness used
/// historically. Useful for `main.rs` sites that only have `dim` at hand.
pub fn load_dataset_config_by_dim(dimension: usize) -> DatasetConfig {
    let name = match dimension {
        32 => "glove25",
        100 => "glove100",
        128 => "sift",
        960 => "gist",
        _ => panic!("Unsupported dimension: {dimension} (no dataset entry in sweep.yaml)"),
    };
    load_dataset_config(name)
}

/// Global DiskANN defaults — only used when no dataset is in scope
/// (e.g. utility scripts). Per-dataset DiskANN config travels on
/// `DatasetConfig::diskann` with field-level fallback to staged → defaults.
pub fn load_diskann_defaults() -> DiskANNConfig {
    let path =
        std::env::var("STAGED_SWEEP_CONFIG").unwrap_or_else(|_| DEFAULT_CONFIG_PATH.to_string());
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

    let staged_d = &root.defaults.staged;
    let ov = ds.staged.as_ref();
    let staged = StagedConfig {
        alpha: ov.and_then(|o| o.alpha).unwrap_or(staged_d.alpha),
        graph_degree: ov
            .and_then(|o| o.graph_degree)
            .unwrap_or(staged_d.graph_degree),
        build_search_list_size: ov
            .and_then(|o| o.build_search_list_size)
            .unwrap_or(staged_d.build_search_list_size),
        max_extra: ov.and_then(|o| o.max_extra).unwrap_or(staged_d.max_extra),
        window_size: ov
            .and_then(|o| o.window_size)
            .unwrap_or(staged_d.window_size),
        metric: Metric::parse(
            ov.and_then(|o| o.metric.as_deref())
                .unwrap_or(staged_d.metric.as_str()),
        ),
    };

    // DiskANN baseline params — per-dataset `diskann:` override block
    // wins; otherwise fall back to staged's params (so QPS comparisons
    // are apples-to-apples on the same graph shape by default).
    // `defaults.diskann` is intentionally NOT consulted here: a global
    // R/L different from the dataset's staged config would silently
    // make the baseline slower, which is exactly the conflation we're
    // trying to avoid. Use the dataset-level override if you want a
    // different DiskANN config than staged.
    let dov = ds.diskann.as_ref();
    let diskann = DiskANNConfig {
        alpha: dov.and_then(|o| o.alpha).unwrap_or(staged.alpha),
        graph_degree: dov
            .and_then(|o| o.graph_degree)
            .unwrap_or(staged.graph_degree),
        build_search_list_size: dov
            .and_then(|o| o.build_search_list_size)
            .unwrap_or(staged.build_search_list_size),
    };

    let bg_raw = ds.base_graph.as_ref();
    let mut source = match bg_raw.and_then(|b| b.source.as_deref()).unwrap_or("rust") {
        "rust" | "Rust" => BaseGraphSource::Rust,
        "parlayann" | "Parlayann" | "PA" | "pa" => BaseGraphSource::Parlayann,
        other => panic!("Unknown base_graph.source: {other} (expected rust|parlayann)"),
    };
    // `staged_file` may reference `${max_extra}` (dataset-context) so
    // the path stays in sync with whatever `staged.max_extra` is set
    // at, plus env-var placeholders like `${PA_ROOT}`. Substitute
    // `${max_extra}` first, then try to resolve env vars — if any
    // env var is unset (typically `PA_ROOT` on a dev box without a
    // ParlayANN checkout), fall back to the Rust in-process build
    // path instead of panicking. Lets every benchmark profile run
    // out of the box without requiring PA_ROOT.
    let mut staged_file: Option<PathBuf> = None;
    if let Some(s) = bg_raw.and_then(|b| b.staged_file.clone()) {
        let max_extra_subbed = s.replace("${max_extra}", &staged.max_extra.to_string());
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
        staged,
        base_graph: BaseGraphConfig {
            source,
            staged_file,
        },
        sweep: SweepConfig {
            search_list_sizes: root.defaults.sweep.search_list_sizes.clone(),
            threads: root.defaults.sweep.threads,
            trials: root.defaults.sweep.trials,
        },
    }
}

/// Replace `${VAR}` occurrences with the corresponding env var, returning
/// `None` if any required var is unset. Lets callers (e.g.
/// `resolve_dataset`) fall back to a Rust in-process build path when
/// `PA_ROOT` isn't available, instead of forcing every profile invocation
/// to set it. Unterminated `${...}` is still a malformed-config panic.
fn try_resolve_placeholders(s: &str) -> Option<String> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let resolved =
            try_resolve_placeholders("${PA_ROOT}/data/glove100/x.staged").unwrap();
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

    #[test]
    fn metric_roundtrip() {
        assert_eq!(Metric::parse("l2"), Metric::L2);
        assert_eq!(Metric::parse("mips"), Metric::Mips);
        assert_eq!(Metric::parse("mips-q"), Metric::MipsQ);
    }
}
