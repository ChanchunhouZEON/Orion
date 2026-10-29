//! Validated configuration; independent of CLI flags, presets, and environment variables.
// Preset/CLI resolution is optional; embedded callers use IndexOptions.
#[cfg(feature = "cli")]
pub mod cli;
#[cfg(feature = "cli")]
pub mod presets;

pub mod options;
pub mod plan;

use crate::data::VectorFormat;
use crate::search::cascade::SearchMetric;
use serde::{Deserialize, Serialize};
use std::{
    hash::{Hash, Hasher},
    path::PathBuf,
};

/// Source used to obtain the base proximity graph.
///
/// The graph can either be constructed by DiskANN native Rust builder or imported
/// from a ParlayANN-generated staged graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum GraphSource {
    /// Construct the graph using DiskANN native Rust implementation with extra
    /// candidates complemented.
    Rust,

    /// Import a graph produced by ParlayANN.
    #[cfg_attr(feature = "cli", value(alias = "pa"))]
    Parlayann,
}

/// User policy, kept separate from the concrete source used in cache identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum GraphSourcePolicy {
    /// Prefer a PA cache/export, then use the Rust builder for f32 storage.
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
    /// SHA-256 of array input; absent for the existing file-identity cache scheme.
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
    pub search_plan: crate::config::plan::SearchPlan,
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
    /// Matching legacy bundled preset, when this configuration is compatible with an
    /// older cache naming scheme.
    #[serde(skip)]
    pub legacy_dataset: Option<String>,
    #[serde(skip)]
    pub legacy_local_pct: usize,
}

impl ResolvedRunConfig {
    /// Only absence allows fallback. Invalid PA caches/exports and permission
    /// errors are surfaced instead of quietly changing the experiment's graph.
    pub fn select_automatic_graph(&mut self, custom_cache_dir: bool) -> Result<(), String> {
        use crate::graph::cache::CachePlan;

        let count = match self.num_points {
            Some(n) => Some(n),
            None => match std::fs::metadata(&self.base) {
                Ok(_) => {
                    Some(crate::data::inspect_vector_file(&self.base, self.base_format)?.count)
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                Err(e) => return Err(format!("{}: {e}", self.base.display())),
            },
        }
        .map(|n| n.min(self.max_points));

        if let Some(n) = count {
            let cache = CachePlan::resolve(self, n).map_err(|error| {
                format!(
                    "automatic PA cache lookup failed in {}: {error}; no Rust graph was substituted",
                    self.cache_dir.display()
                )
            })?;
            if cache.is_hit {
                self.graph_selection_reason = "PA cache available";
                return Ok(());
            }
        }
        if let Some(path) = CachePlan::available_staged_export(self)? {
            if let Some(n) = count {
                CachePlan::resolve_staged_export(self, n)?;
            }
            self.staged_file = Some(path);
            self.graph_selection_reason = "PA staged export available";
            return Ok(());
        }
        if self.search_plan().storage().kind == VectorStorageKind::U8 {
            // Offline preflight can still estimate this configuration. Execution
            // will report the missing PA input before allocating the dataset.
            self.graph_selection_reason =
                "native u8 requires a PA cache or staged export; no Rust fallback";
            return Ok(());
        }

        self.graph_source = GraphSource::Rust;
        if !custom_cache_dir {
            self.cache_dir = PathBuf::from("cache/orion");
        }
        self.cache_namespace = self.compute_cache_namespace();
        self.graph_selection_reason = if count
            .map(|n| CachePlan::resolve(self, n).map(|cache| cache.is_hit))
            .transpose()?
            .unwrap_or(false)
        {
            "PA unavailable; Rust cache available"
        } else {
            "PA unavailable; Rust build required"
        };
        Ok(())
    }

    pub fn search_plan(&self) -> crate::config::plan::SearchPlan {
        self.search_plan
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
    pub fn compute_cache_namespace(&self) -> String {
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
            self.base_content_id
                .as_ref()
                .map(|id| (PathBuf::from("/__orion_array_v1__").join(id), None, None))
                .unwrap_or_else(|| base_file_identity(&self.base)),
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
        if self.search_plan().storage().kind == VectorStorageKind::U8 {
            "native-u8-v1".hash(&mut hash);
        }
        format!("v3_{:016x}", hash.finish())
    }
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
