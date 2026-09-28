/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Cache lookup is independent of the availability of the original graph export.
use super::config::{GraphSource, ResolvedRunConfig};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

#[allow(dead_code)]
#[derive(Debug)]
pub struct CachePlan {
    pub path: PathBuf,
    pub hit: bool,
    pub legacy: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheManifest {
    version: u32,
    namespace: String,
    num_points: usize,
    dimension: usize,
    imported_from: Option<SourceFile>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SourceFile {
    path: PathBuf,
    size: u64,
    modified_ns: Option<u128>,
}

#[allow(dead_code)]
fn absolute_path(path: &Path) -> Result<PathBuf, String> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
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

impl SourceFile {
    fn read(path: &Path) -> Result<Self, String> {
        let metadata = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if !metadata.is_file() {
            return Err(format!("{} is not a file", path.display()));
        }
        Ok(Self {
            path: absolute_path(path)?,
            size: metadata.len(),
            modified_ns: metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos()),
        })
    }

    fn matches_explicit(&self, path: &Path) -> Result<bool, String> {
        if self.path != absolute_path(path)? {
            return Ok(false);
        }
        // Reusing an explicitly named, since-removed export is still a cache hit.
        match std::fs::metadata(path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(true),
            Err(e) => Err(e.to_string()),
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

fn validate_headers(path: &Path, num_points: usize, degree: u32) -> Result<(), String> {
    let mut meta = [0; 8];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut meta))
        .map_err(|e| e.to_string())?;
    let version = u32::from_le_bytes(meta[..4].try_into().unwrap());
    let entry = u32::from_le_bytes(meta[4..].try_into().unwrap()) as usize;
    if version != 5 || entry >= num_points {
        return Err(format!(
            "invalid cache metadata: version={version}, entry={entry}, N={num_points}"
        ));
    }
    let graph = path.with_extension("pgraph");
    let mut file = std::fs::File::open(&graph).map_err(|e| e.to_string())?;
    let mut header = [0; 16];
    file.read_exact(&mut header).map_err(|e| e.to_string())?;
    let word = |i: usize| u32::from_le_bytes(header[i * 4..i * 4 + 4].try_into().unwrap());
    let n = word(0) as usize;
    let r = word(1);
    let stride = word(2) as u64;
    let expected_bytes = (n as u64)
        .checked_mul(stride)
        .and_then(|v| v.checked_mul(4))
        .and_then(|v| v.checked_add(16))
        .ok_or("cache size overflow")?;
    if n != num_points
        || r != degree
        || stride < r as u64 + 4
        || file.metadata().map_err(|e| e.to_string())?.len() != expected_bytes
    {
        return Err(format!("cache header/size mismatch: N={n}, R={r}, stride={stride}; expected N={num_points}, R={degree}"));
    }
    Ok(())
}

fn check_manifest(path: &Path, config: &ResolvedRunConfig, n: usize) -> Result<(), String> {
    let metadata: CacheManifest =
        serde_json::from_slice(&std::fs::read(manifest_path(path)).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if metadata.version != 1
        || metadata.namespace != config.cache_namespace
        || metadata.num_points != n
        || metadata.dimension != config.dimension
    {
        return Err("cache provenance does not match the current dataset/configuration".into());
    }
    if config.graph_source == GraphSource::Parlayann && config.staged_file_explicit {
        let requested = config
            .staged_file
            .as_deref()
            .ok_or("missing explicit staged_file")?;
        let matches = metadata
            .imported_from
            .as_ref()
            .map(|source| source.matches_explicit(requested))
            .transpose()?
            .unwrap_or(false);
        if !matches {
            return Err("explicit staged_file differs from the cached source (or provenance is unknown); choose another --cache-dir to import it without overwriting this cache".into());
        }
    }
    Ok(())
}

impl CachePlan {
    pub fn select(config: &ResolvedRunConfig, n: usize) -> Result<Self, String> {
        let path = config
            .cache_dir
            .join(format!("{}_n{n}.bin", config.cache_namespace));
        let graph = path.with_extension("pgraph");
        if path.exists() || graph.exists() {
            validate_headers(&path, n, config.graph_degree)?;
            check_manifest(&path, config, n)?;
            return Ok(Self {
                path,
                hit: true,
                legacy: false,
            });
        }
        if let Some(name) = &config.legacy_dataset {
            let alpha = format!("{:.2}", config.alpha).replace('.', "_");
            let suffix = if config.graph_source == GraphSource::Parlayann {
                format!("_pct{}", config.legacy_local_pct)
            } else {
                String::new()
            };
            let legacy = config.cache_dir.join(format!(
                "{name}_n{n}_r{}_l{}_a{alpha}_ex{}{suffix}.bin",
                config.graph_degree, config.build_l, config.max_extra
            ));
            if legacy.is_file() && legacy.with_extension("pgraph").is_file() {
                // Old caches carry no export identity. An explicit alternative export must
                // not be ignored merely because a name-based legacy cache happens to exist.
                if config.staged_file_explicit && !manifest_path(&legacy).is_file() {
                    log::warn!("Legacy cache has no source provenance; importing the explicitly selected export into a new cache");
                } else {
                    validate_headers(&legacy, n, config.graph_degree)?;
                    if manifest_path(&legacy).is_file() {
                        check_manifest(&legacy, config, n)?;
                    }
                    log::info!(
                        "Reusing preset-compatible legacy cache {}",
                        legacy.display()
                    );
                    return Ok(Self {
                        path: legacy,
                        hit: true,
                        legacy: true,
                    });
                }
            }
        }
        Ok(Self {
            path,
            hit: false,
            legacy: false,
        })
    }

    /// Called only on a PA cache miss. Merely resolving a candidate path performs no IO.
    pub fn source_for_import(config: &ResolvedRunConfig, n: usize) -> Result<PathBuf, String> {
        let mut path = config.staged_file.clone().ok_or(
            "ParlayANN cache miss: provide --staged-file or ORION_STAGED_FILE, or set PA_ROOT for the YAML export path"
        )?;
        // The prep script appends _pct60; also accept older YAML paths without that suffix.
        if !config.staged_file_explicit && !path.is_file() {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                let alternate =
                    path.with_file_name(format!("{stem}_pct{}.staged", config.legacy_local_pct));
                if alternate.is_file() {
                    path = alternate;
                }
            }
        }
        let mut file = std::fs::File::open(&path).map_err(|e| format!(
            "ParlayANN cache miss; cannot import {}: {e}. Set --staged-file/ORION_STAGED_FILE or PA_ROOT (default ../ParlayANN). No Rust graph was substituted.",
            path.display()
        ))?;
        let mut header = [0; 24];
        file.read_exact(&mut header)
            .map_err(|e| format!("invalid staged header: {e}"))?;
        let word = |i: usize| u32::from_le_bytes(header[i * 4..i * 4 + 4].try_into().unwrap());
        if word(0) != 0x53544147 || word(1) != 3 || word(2) as usize != n {
            return Err(format!("staged header must be STAG v3 with N={n}"));
        }
        Ok(path)
    }

    pub fn record(
        &self,
        config: &ResolvedRunConfig,
        n: usize,
        source: Option<&Path>,
    ) -> Result<(), String> {
        let path = manifest_path(&self.path);
        if self.hit && path.is_file() {
            return Ok(());
        }
        let manifest = CacheManifest {
            version: 1,
            namespace: config.cache_namespace.clone(),
            num_points: n,
            dimension: config.dimension,
            imported_from: source.map(SourceFile::read).transpose()?,
        };
        let temporary = path.with_extension(format!("json.{}.tmp", std::process::id()));
        let bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
        std::fs::write(&temporary, bytes).map_err(|e| e.to_string())?;
        std::fs::rename(&temporary, &path).map_err(|e| e.to_string())?;
        if self.legacy {
            log::warn!("Legacy headers were validated and current input identity recorded; historical export provenance is unavailable");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::Args;
    use clap::Parser;
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
        let plan = CachePlan::select(&c, 1).unwrap();
        assert!(!plan.hit);
        write_cache(&plan.path, 1, 4);
        let export = dir.0.join("source.staged");
        std::fs::write(&export, b"fixture").unwrap();
        plan.record(&c, 1, Some(&export)).unwrap();
        std::fs::remove_file(&export).unwrap();
        assert!(CachePlan::select(&c, 1).unwrap().hit);
        c.staged_file_explicit = true;
        c.staged_file = Some(export);
        assert!(CachePlan::select(&c, 1).unwrap().hit);
        c.staged_file = Some(dir.0.join("different.staged"));
        assert!(CachePlan::select(&c, 1).is_err());
    }

    #[test]
    fn legacy_lookup_is_opt_in_and_checks_headers() {
        let dir = Temp::new();
        let mut c = config(&dir.0);
        let legacy = dir.0.join("fixture_n1_r4_l100_a1_20_ex16_pct60.bin");
        write_cache(&legacy, 1, 4);
        assert!(!CachePlan::select(&c, 1).unwrap().hit);
        c.legacy_dataset = Some("fixture".into());
        let plan = CachePlan::select(&c, 1).unwrap();
        assert!(plan.hit && plan.legacy);
        plan.record(&c, 1, None).unwrap();
        assert!(CachePlan::select(&c, 1).unwrap().hit);
        write_cache(&legacy, 2, 4);
        assert!(CachePlan::select(&c, 1).is_err());
    }

    #[test]
    fn explicit_export_does_not_silently_reuse_unverified_legacy() {
        let dir = Temp::new();
        let mut c = config(&dir.0);
        c.legacy_dataset = Some("fixture".into());
        write_cache(&dir.0.join("fixture_n1_r4_l100_a1_20_ex16_pct60.bin"), 1, 4);
        c.staged_file_explicit = true;
        c.staged_file = Some(dir.0.join("new.staged"));
        assert!(!CachePlan::select(&c, 1).unwrap().hit);
    }

    #[test]
    fn pa_miss_requires_export_and_never_changes_graph_source() {
        let dir = Temp::new();
        let c = config(&dir.0);
        assert!(!CachePlan::select(&c, 1).unwrap().hit);
        assert!(CachePlan::source_for_import(&c, 1).is_err());
        assert_eq!(c.graph_source, GraphSource::Parlayann);
    }
}
