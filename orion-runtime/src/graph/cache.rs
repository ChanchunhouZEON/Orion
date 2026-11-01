/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Cache lookup is independent of the availability of the original graph export.
use crate::config::{GraphSource, ResolvedRunConfig};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

/// Selected metadata file and whether its graph/provenance have been validated.
/// A miss names the destination for a newly built/imported index.
#[allow(dead_code)]
#[derive(Debug)]
pub struct CachePlan {
    pub metadata_path: PathBuf,
    pub is_hit: bool,
    pub is_legacy: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheManifest {
    version: u32,
    namespace: String,
    num_points: usize,
    dimension: usize,
    imported_from: Option<SourceFileIdentity>,
    /// Older manifests identify the source through their namespace only.
    #[serde(default)]
    graph_source: Option<GraphSource>,
    #[serde(default)]
    builder_recipe: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SourceFileIdentity {
    path: PathBuf,
    size: u64,
    modified_ns: Option<u128>,
}

/// Normalize lexically without requiring the export to still exist on disk.
#[allow(dead_code)]
fn normalize_absolute_path(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir().map_err(Error::from)?.join(path)
    };

    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    Ok(normalized)
}

impl SourceFileIdentity {
    fn read(path: &Path) -> Result<Self> {
        let metadata = std::fs::metadata(path).map_err(|e| {
            let message = format!("{}: {e}", path.display());
            Error::from(e).with_message(message)
        })?;
        if !metadata.is_file() {
            return Err(Error::from(format!("{} is not a file", path.display())));
        }
        Ok(Self {
            path: normalize_absolute_path(path)?,
            size: metadata.len(),
            modified_ns: metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos()),
        })
    }

    fn matches_requested_export(&self, path: &Path) -> Result<bool> {
        if self.path != normalize_absolute_path(path)? {
            return Ok(false);
        }

        // Reusing an explicitly named, since-removed export is still a cache hit.
        match std::fs::metadata(path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(true),
            Err(e) => Err(e.into()),
            Ok(_) => {
                let current = Self::read(path)?;
                Ok(current.size == self.size && current.modified_ns == self.modified_ns)
            }
        }
    }
}

fn manifest_path(path: &Path) -> PathBuf {
    path.with_extension("cache.json")
}

fn validate_headers(path: &Path, num_points: usize, degree: u32) -> Result<()> {
    let mut metadata_bytes = [0; 8];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut metadata_bytes))
        .map_err(Error::from)?;
    let version = u32::from_le_bytes(metadata_bytes[..4].try_into().unwrap());
    let entry = u32::from_le_bytes(metadata_bytes[4..].try_into().unwrap()) as usize;
    // V6 adds logical dataset shape after the unchanged eight-byte prefix.
    // The index loader checks that shape before loading the graph payload;
    // this lightweight lookup also validates the existing provenance manifest.
    if !matches!(version, 5 | 6) || entry >= num_points {
        return Err(Error::from(format!(
            "invalid cache metadata: version={version}, entry={entry}, N={num_points}"
        )));
    }

    let graph = path.with_extension("pgraph");
    let mut file = std::fs::File::open(&graph).map_err(Error::from)?;
    let mut header = [0; 16];
    file.read_exact(&mut header).map_err(Error::from)?;
    let word = |i: usize| u32::from_le_bytes(header[i * 4..i * 4 + 4].try_into().unwrap());
    let stored_points = word(0) as usize;
    let stored_degree = word(1);
    let stride_words = word(2) as u64;

    // The header is followed by fixed-stride node records, measured in u32 words.
    let expected_bytes = (stored_points as u64)
        .checked_mul(stride_words)
        .and_then(|v| v.checked_mul(4))
        .and_then(|v| v.checked_add(16))
        .ok_or("cache size overflow")?;
    if stored_points != num_points
        || stored_degree != degree
        || stride_words < stored_degree as u64 + 4
        || file.metadata().map_err(Error::from)?.len() != expected_bytes
    {
        return Err(Error::cache(format!(
            "cache header/size mismatch: N={stored_points}, R={stored_degree}, \
             stride={stride_words}; expected N={num_points}, R={degree}"
        )));
    }
    Ok(())
}

fn validate_manifest(path: &Path, config: &ResolvedRunConfig, num_points: usize) -> Result<()> {
    let manifest_bytes = std::fs::read(manifest_path(path)).map_err(Error::from)?;
    let metadata: CacheManifest = serde_json::from_slice(&manifest_bytes).map_err(Error::from)?;
    if metadata.version != 1
        || metadata.namespace != config.cache_namespace
        || metadata.num_points != num_points
        || metadata.dimension != config.dimension
        || metadata
            .graph_source
            .is_some_and(|source| source != config.graph_source)
    {
        return Err(Error::cache(
            "cache provenance does not match the current dataset/configuration",
        )
        .at("load_cache")
        .with_detail("path", path.to_string_lossy().into_owned()));
    }

    if metadata
        .builder_recipe
        .as_deref()
        .is_some_and(|recipe| recipe != super::PARLAY_RECIPE)
    {
        return Err(Error::cache("unsupported ParlayANN builder recipe").at("load_cache"));
    }
    if config.graph_source == GraphSource::Parlayann && config.staged_file_explicit {
        let requested = config
            .staged_file
            .as_deref()
            .ok_or("missing explicit staged_file")?;
        let matches = metadata
            .imported_from
            .as_ref()
            .map(|source| source.matches_requested_export(requested))
            .transpose()?
            .unwrap_or(false);
        if !matches {
            return Err(
                "explicit staged_file differs from the cached source (or provenance is unknown); \
                 choose another --cache-dir to import it without overwriting this cache"
                    .into(),
            );
        }
    }
    Ok(())
}

fn legacy_cache_path(config: &ResolvedRunConfig, name: &str, num_points: usize) -> PathBuf {
    let alpha = format!("{:.2}", config.alpha).replace('.', "_");
    let suffix = if config.graph_source == GraphSource::Parlayann {
        format!("_pct{}", config.legacy_local_pct)
    } else {
        String::new()
    };
    config.cache_dir.join(format!(
        "{name}_n{num_points}_r{}_l{}_a{alpha}_ex{}{suffix}.bin",
        config.graph_degree, config.build_l, config.max_extra
    ))
}

impl CachePlan {
    pub fn resolve(config: &ResolvedRunConfig, num_points: usize) -> Result<Self> {
        let path = config
            .cache_dir
            .join(format!("{}_n{num_points}.bin", config.cache_namespace));

        let graph = path.with_extension("pgraph");

        // Either file means a cache was started. Reject incomplete pairs instead
        // of treating them as misses and overwriting the remaining file.
        if path.try_exists().map_err(|e| {
            let message = format!("{}: {e}", path.display());
            Error::from(e).with_message(message)
        })? || graph.try_exists().map_err(|e| {
            let message = format!("{}: {e}", graph.display());
            Error::from(e).with_message(message)
        })? {
            validate_headers(&path, num_points, config.graph_degree)?;
            validate_manifest(&path, config, num_points)?;
            return Ok(Self {
                metadata_path: path,
                is_hit: true,
                is_legacy: false,
            });
        }

        if let Some(name) = &config.legacy_dataset {
            let legacy = legacy_cache_path(config, name, num_points);
            if legacy.try_exists().map_err(Error::from)?
                || legacy
                    .with_extension("pgraph")
                    .try_exists()
                    .map_err(Error::from)?
            {
                // Old caches carry no export identity. An explicit alternative export must
                // not be ignored merely because a name-based legacy cache happens to exist.
                if config.staged_file_explicit && !manifest_path(&legacy).is_file() {
                    log::warn!(
                        "Legacy cache has no source provenance; \
                         importing the explicitly selected export into a new cache"
                    );
                } else {
                    validate_headers(&legacy, num_points, config.graph_degree)?;
                    if manifest_path(&legacy).is_file() {
                        validate_manifest(&legacy, config, num_points)?;
                    }
                    log::info!(
                        "Reusing preset-compatible legacy cache {}",
                        legacy.display()
                    );
                    return Ok(Self {
                        metadata_path: legacy,
                        is_hit: true,
                        is_legacy: true,
                    });
                }
            }
        }

        Ok(Self {
            metadata_path: path,
            is_hit: false,
            is_legacy: false,
        })
    }

    /// Locate the default export (including the historical `_pct60` suffix).
    /// Only NotFound means unavailable; unreadable/non-file inputs are errors.
    pub fn available_staged_export(config: &ResolvedRunConfig) -> Result<Option<PathBuf>> {
        let Some(path) = &config.staged_file else {
            return Ok(None);
        };
        let mut candidates = vec![path.clone()];
        if !config.staged_file_explicit {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                candidates.push(
                    path.with_file_name(format!("{stem}_pct{}.staged", config.legacy_local_pct)),
                );
            }
        }
        for path in candidates {
            match std::fs::metadata(&path) {
                Ok(meta) if meta.is_file() => return Ok(Some(path)),
                Ok(_) => {
                    return Err(Error::from(format!(
                        "{} is not a staged file",
                        path.display()
                    )));
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(Error::from(format!("{}: {e}", path.display()))),
            }
        }
        Ok(None)
    }

    /// Resolve import versus in-memory construction before allocating the base.
    /// Explicit missing exports remain errors, even when the builder is present.
    pub fn parlay_input(config: &ResolvedRunConfig, num_points: usize) -> Result<Option<PathBuf>> {
        if config.staged_file_explicit || Self::available_staged_export(config)?.is_some() {
            return Self::resolve_staged_export(config, num_points).map(Some);
        }
        if cfg!(feature = "parlay") {
            return Ok(None);
        }
        Err(Error::unsupported("ParlayANN construction is unavailable; enable the parlay feature or provide a staged export")
            .at("build_graph").with_detail("backend", "parlayann"))
    }

    /// Resolve and validate the STAG header on a cache miss.
    /// Cache hits skip this IO: the original export may have been removed.
    pub fn resolve_staged_export(config: &ResolvedRunConfig, num_points: usize) -> Result<PathBuf> {
        let path = Self::available_staged_export(config)?
            .or_else(|| config.staged_file.clone())
            .ok_or("ParlayANN cache miss: provide --staged-file or ORION_STAGED_FILE, or set PA_ROOT for the YAML export path. No Rust graph was substituted.")?;

        let mut file = std::fs::File::open(&path).map_err(|e| {
            let message = format!(
                "ParlayANN cache miss; cannot import {}: {e}. Set --staged-file/ORION_STAGED_FILE or PA_ROOT (default ../ParlayANN). No Rust graph was substituted.", path.display());
            Error::io("open", &path, e).with_message(message)
        })?;
        let mut header = [0; 24];
        file.read_exact(&mut header).map_err(|e| {
            let message = format!("invalid staged header: {e}");
            Error::from(e).with_message(message)
        })?;

        // STAG v3 starts with magic, version, and point count. The streaming
        // importer validates capacities and adjacency records when it loads them.
        let word = |i: usize| u32::from_le_bytes(header[i * 4..i * 4 + 4].try_into().unwrap());
        if word(0) != 0x53544147 || word(1) != 3 || word(2) as usize != num_points {
            return Err(Error::invalid_data(format!(
                "staged header must be STAG v3 with N={num_points}"
            )));
        }
        Ok(path)
    }

    /// Publish provenance after graph and metadata have been saved successfully.
    /// Missing manifests are rejected on the next lookup rather than guessed.
    pub fn write_manifest(
        &self,
        config: &ResolvedRunConfig,
        num_points: usize,
        source: Option<&Path>,
    ) -> Result<()> {
        let path = manifest_path(&self.metadata_path);
        if self.is_hit && path.is_file() {
            return Ok(());
        }

        let manifest = CacheManifest {
            version: 1,
            namespace: config.cache_namespace.clone(),
            num_points: num_points,
            dimension: config.dimension,
            imported_from: source.map(SourceFileIdentity::read).transpose()?,
            graph_source: Some(config.graph_source),
            builder_recipe: if !self.is_hit
                && source.is_none()
                && config.graph_source == GraphSource::Parlayann
            {
                Some(super::PARLAY_RECIPE.into())
            } else {
                None
            },
        };

        // Publish with a same-directory rename so readers never see partial JSON.
        let temporary = path.with_extension(format!("json.{}.tmp", std::process::id()));
        let bytes = serde_json::to_vec_pretty(&manifest).map_err(Error::from)?;
        std::fs::write(&temporary, bytes).map_err(Error::from)?;
        std::fs::rename(&temporary, &path).map_err(Error::from)?;
        if self.is_legacy {
            log::warn!(
                "Legacy headers were validated and current input identity recorded; \
                 historical export provenance is unavailable"
            );
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cli"))]
mod tests {
    use super::*;
    use crate::config::cli::Args;
    use crate::config::{GraphSource, ResolvedRunConfig};
    use clap::Parser;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            static ID: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "orion-cache-test-{}-{}",
                std::process::id(),
                ID.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn config(dir: &Path) -> ResolvedRunConfig {
        let mut c = Args::try_parse_from([
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
            "--graph-degree",
            "4",
        ])
        .unwrap()
        .resolve()
        .unwrap();
        c.cache_dir = dir.to_owned();
        c.staged_file = None;
        c.staged_file_explicit = false;
        c
    }
    fn write_cache(path: &Path, n: u32, r: u32) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let bytes = [5u32, 0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        std::fs::write(path, bytes).unwrap();
        let mut graph = [n, r, 16, 0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        graph.resize(16 + n as usize * 16 * 4, 0);
        std::fs::write(path.with_extension("pgraph"), graph).unwrap();
    }

    #[test]
    fn cached_pa_graph_survives_export_removal() {
        let dir = Temp::new();
        let mut c = config(&dir.0);
        let plan = CachePlan::resolve(&c, 1).unwrap();
        assert!(!plan.is_hit);
        write_cache(&plan.metadata_path, 1, 4);
        let export = dir.0.join("source.staged");
        std::fs::write(&export, b"fixture").unwrap();
        plan.write_manifest(&c, 1, Some(&export)).unwrap();
        std::fs::remove_file(&export).unwrap();
        assert!(CachePlan::resolve(&c, 1).unwrap().is_hit);
        c.staged_file_explicit = true;
        c.staged_file = Some(export);
        assert!(CachePlan::resolve(&c, 1).unwrap().is_hit);
        c.staged_file = Some(dir.0.join("different.staged"));
        assert!(CachePlan::resolve(&c, 1).is_err());
    }

    #[test]
    fn legacy_lookup_is_opt_in_and_checks_headers() {
        let dir = Temp::new();
        let mut c = config(&dir.0);
        let legacy = dir.0.join("fixture_n1_r4_l100_a1_20_ex16_pct60.bin");
        write_cache(&legacy, 1, 4);
        assert!(!CachePlan::resolve(&c, 1).unwrap().is_hit);
        c.legacy_dataset = Some("fixture".into());
        let plan = CachePlan::resolve(&c, 1).unwrap();
        assert!(plan.is_hit && plan.is_legacy);
        plan.write_manifest(&c, 1, None).unwrap();
        assert!(CachePlan::resolve(&c, 1).unwrap().is_hit);
        write_cache(&legacy, 2, 4);
        assert!(CachePlan::resolve(&c, 1).is_err());
    }

    #[test]
    fn explicit_export_does_not_silently_reuse_unverified_legacy() {
        let dir = Temp::new();
        let mut c = config(&dir.0);
        c.legacy_dataset = Some("fixture".into());
        write_cache(&dir.0.join("fixture_n1_r4_l100_a1_20_ex16_pct60.bin"), 1, 4);
        c.staged_file_explicit = true;
        c.staged_file = Some(dir.0.join("new.staged"));
        assert!(!CachePlan::resolve(&c, 1).unwrap().is_hit);
    }

    #[test]
    fn pa_miss_requires_export_and_never_changes_graph_source() {
        let dir = Temp::new();
        let c = config(&dir.0);
        assert!(!CachePlan::resolve(&c, 1).unwrap().is_hit);
        assert!(CachePlan::resolve_staged_export(&c, 1).is_err());
        assert_eq!(c.graph_source, GraphSource::Parlayann);
    }
}
