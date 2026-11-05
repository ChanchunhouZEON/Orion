/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Cache lookup is independent of the availability of the original graph export.
//! CachePlan selects a Generation containing a CacheManifest and its exact bytes.
//! Those records precede lookup/publication operations; validation helpers follow.
use super::staged::SourceFileIdentity;
use crate::config::{GraphSource, ResolvedRunConfig};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Selected metadata file and whether its graph/provenance have been validated.
/// A miss names the destination for a newly built/imported index.
#[derive(Debug)]
pub struct CachePlan {
    pub metadata_path: PathBuf,
    pub(super) selected: Option<Generation>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CacheManifest {
    pub(crate) graph_sha256: String,
    pub(crate) metadata_sha256: String,
    namespace: String,
    num_points: usize,
    dimension: usize,
    imported_from: Option<SourceFileIdentity>,
    graph_source: GraphSource,
    builder_recipe: Option<String>,
}

/// Keep the exact working record used to select and verify a resident graph.
#[derive(Debug)]
pub(crate) struct Generation {
    pub(crate) manifest: CacheManifest,
    pub(crate) encoded: Vec<u8>,
}

impl CachePlan {
    pub(crate) fn is_hit(&self) -> bool {
        self.selected.is_some()
    }

    fn generation(&self) -> Result<&Generation> {
        self.selected
            .as_ref()
            .ok_or_else(|| Error::invalid_state("no selected graph generation"))
    }
    pub(crate) fn objects(&self) -> Result<PathBuf> {
        crate::storage::object_dir(self.metadata_path.parent().unwrap_or(Path::new(".")))
    }

    pub(crate) fn graph_path(&self) -> Result<PathBuf> {
        crate::storage::object(
            &self.objects()?,
            &self.generation()?.manifest.graph_sha256,
            "pgraph",
        )
    }

    pub(crate) fn lookup_key(config: &ResolvedRunConfig, num_points: usize) -> PathBuf {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(config.cache_namespace.as_bytes());
        hash.update((num_points as u64).to_le_bytes());
        config.cache_dir.join(format!("{:x}.bin", hash.finalize()))
    }

    pub fn resolve(config: &ResolvedRunConfig, num_points: usize) -> Result<Self> {
        Self::plan(config, num_points, true)
    }

    pub(crate) fn inspect(config: &ResolvedRunConfig, num_points: usize) -> Result<Self> {
        Self::plan(config, num_points, false)
    }

    fn plan(config: &ResolvedRunConfig, num_points: usize, verify_source: bool) -> Result<Self> {
        let path = Self::lookup_key(config, num_points);
        let key = manifest_path(&path);
        let selected = match std::fs::File::open(&key) {
            Ok(file) => {
                let encoded = crate::storage::read_bounded_file(file, &key, 1024 * 1024)?;
                Some(Generation {
                    manifest: serde_json::from_slice(&encoded)?,
                    encoded,
                })
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(Error::io("open cache record", &key, e)),
        };
        let plan = Self {
            selected,
            metadata_path: path,
        };
        if plan.metadata_path.try_exists()?
            || plan.metadata_path.with_extension("pgraph").try_exists()?
        {
            return Err(Error::cache(
                "legacy graph cache; use a fresh working directory",
            ));
        }
        if let Some(generation) = &plan.selected {
            let record = &generation.manifest;
            validate_manifest(
                record,
                &plan.metadata_path,
                config,
                num_points,
                verify_source,
            )?;
            let dir = plan.objects()?;
            validate_headers(
                &crate::storage::object(&dir, &record.metadata_sha256, "bin")?,
                &crate::storage::object(&dir, &record.graph_sha256, "pgraph")?,
                num_points,
                config.graph_degree,
            )?;
        }
        Ok(plan)
    }

    /// Publish provenance after graph and metadata have been saved successfully.
    /// Missing manifests are rejected on the next lookup rather than guessed.
    pub(super) fn write_manifest(
        &self,
        config: &ResolvedRunConfig,
        num_points: usize,
        source: Option<SourceFileIdentity>,
        graph_sha256: &str,
        metadata_sha256: &str,
    ) -> Result<Generation> {
        let path = manifest_path(&self.metadata_path);

        let manifest = CacheManifest {
            graph_sha256: graph_sha256.into(),
            metadata_sha256: metadata_sha256.into(),
            namespace: config.cache_namespace.clone(),
            num_points: num_points,
            dimension: config.dimension,

            graph_source: config.graph_source,
            builder_recipe: if !self.is_hit()
                && source.is_none()
                && config.graph_source == GraphSource::Parlayann
            {
                Some(super::PARLAY_RECIPE.into())
            } else {
                None
            },
            imported_from: source,
        };

        // Sync the complete record before linking it into the working lookup.
        use std::io::Write;
        std::fs::create_dir_all(&config.cache_dir)?;
        let (temporary, mut file) = crate::storage::PendingFile::create(&path)?;
        let encoded = serde_json::to_vec_pretty(&manifest)?;
        file.write_all(&encoded)?;
        file.sync_all()?;
        drop(file);
        // A concurrent builder may have produced a different graph for this recipe.
        // Never replace its record after another reader has selected that generation.
        std::fs::hard_link(&temporary.0, &path)?;
        crate::storage::sync_dir(&config.cache_dir)?;
        Ok(Generation { manifest, encoded })
    }
}

fn manifest_path(path: &Path) -> PathBuf {
    path.with_extension("cache.json")
}

fn validate_headers(path: &Path, graph: &Path, num_points: usize, degree: u32) -> Result<()> {
    let mut metadata_bytes = [0; 8];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut metadata_bytes))
        .map_err(Error::from)?;
    let version = u32::from_le_bytes(metadata_bytes[..4].try_into().unwrap());
    let entry = u32::from_le_bytes(metadata_bytes[4..].try_into().unwrap()) as usize;
    // V6 adds logical dataset shape after the unchanged eight-byte prefix.
    // The index loader checks that shape before loading the graph payload;
    // this lightweight lookup also validates the existing provenance manifest.
    if version != 6 || entry >= num_points {
        return Err(Error::from(format!(
            "invalid cache metadata: version={version}, entry={entry}, N={num_points}"
        )));
    }

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

fn validate_manifest(
    metadata: &CacheManifest,
    path: &Path,
    config: &ResolvedRunConfig,
    num_points: usize,
    verify_source: bool,
) -> Result<()> {
    if metadata.namespace != config.cache_namespace
        || metadata.num_points != num_points
        || metadata.dimension != config.dimension
        || metadata.graph_source != config.graph_source
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
    if verify_source && config.graph_source == GraphSource::Parlayann && config.staged_file_explicit
    {
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

#[cfg(all(test, feature = "cli"))]
mod tests {
    use super::*;
    use crate::config::{GraphSource, ResolvedRunConfig};
    use crate::driver::Args;
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
        let bytes = [6u32, 0]
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
        assert!(!plan.is_hit());
        let path = dir.0.join("fixture");
        write_cache(&path, 1, 4);
        let store = plan.objects().unwrap();
        let put = |ext| {
            crate::storage::put(&store, ext, |w| {
                std::io::Write::write_all(w, &std::fs::read(path.with_extension(ext))?)?;
                Ok(())
            })
            .unwrap()
            .1
        };
        std::fs::rename(&path, path.with_extension("bin")).unwrap();
        let graph = put("pgraph");
        let meta = put("bin");
        let export = dir.0.join("source.staged");
        std::fs::write(&export, b"fixture").unwrap();
        plan.write_manifest(
            &c,
            1,
            Some(
                SourceFileIdentity::from_file(&export, std::fs::File::open(&export).unwrap())
                    .unwrap(),
            ),
            &graph,
            &meta,
        )
        .unwrap();
        std::fs::remove_file(&export).unwrap();
        assert!(CachePlan::resolve(&c, 1).unwrap().is_hit());
        c.staged_file_explicit = true;
        c.staged_file = Some(export);
        assert!(CachePlan::resolve(&c, 1).unwrap().is_hit());
        c.staged_file = Some(dir.0.join("different.staged"));
        assert!(CachePlan::resolve(&c, 1).is_err());
    }

    #[test]
    fn explicit_export_does_not_silently_reuse_unverified_legacy() {
        let dir = Temp::new();
        let mut c = config(&dir.0);
        write_cache(&dir.0.join("fixture_n1_r4_l100_a1_20_ex16_pct60.bin"), 1, 4);
        c.staged_file_explicit = true;
        c.staged_file = Some(dir.0.join("new.staged"));
        assert!(!CachePlan::resolve(&c, 1).unwrap().is_hit());
    }

    #[test]
    fn pa_miss_requires_export_and_never_changes_graph_source() {
        let dir = Temp::new();
        let c = config(&dir.0);
        assert!(!CachePlan::resolve(&c, 1).unwrap().is_hit());
        assert!(crate::graph::staged::resolve_staged_export(&c, 1).is_err());
        assert_eq!(c.graph_source, GraphSource::Parlayann);
    }
}
