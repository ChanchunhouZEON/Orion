/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Select, identify and stream a ParlayANN staged export.
use crate::{config::ResolvedRunConfig, data::BaseElement, Error, Result};
use orion_ann_core::Orion;
use orion_ann_diskann::model::PaddedDataset;
use serde::{Deserialize, Serialize};
use std::{
    io::Read,
    path::{Component, Path, PathBuf},
};
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct SourceFileIdentity {
    path: PathBuf,
    size: u64,
    sha256: String,
}

/// Normalize lexically without requiring the export to still exist on disk.
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
    pub(super) fn from_stream(path: &Path, size: u64, sha256: String) -> Result<Self> {
        Ok(Self {
            path: normalize_absolute_path(path)?,
            size,
            sha256,
        })
    }

    pub(super) fn from_file(path: &Path, file: std::fs::File) -> Result<Self> {
        if !file.metadata()?.is_file() {
            return Err(Error::invalid_data(format!(
                "{} is not a file",
                path.display()
            )));
        }
        let (bytes, digest) = crate::integrity::hash_file(file)?;
        Self::from_stream(path, bytes, digest)
    }

    pub(super) fn matches_requested_export(&self, path: &Path) -> Result<bool> {
        match std::fs::File::open(path) {
            Ok(file) => {
                let current = Self::from_file(path, file)?;
                Ok(current.size == self.size && current.sha256 == self.sha256)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Ok(self.path == normalize_absolute_path(path)?)
            }
            Err(e) => Err(Error::io("open staged export", path, e)),
        }
    }
}

/// Locate the default export (including the historical `_pct60` suffix).
/// Only NotFound means unavailable; unreadable/non-file inputs are errors.
pub(crate) fn available_staged_export(config: &ResolvedRunConfig) -> Result<Option<PathBuf>> {
    let Some(path) = &config.staged_file else {
        return Ok(None);
    };
    let mut candidates = vec![path.clone()];
    if !config.staged_file_explicit {
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            candidates.push(path.with_file_name(format!("{stem}_pct{}.staged", config.local_pct)));
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
pub(crate) fn parlay_input(
    config: &ResolvedRunConfig,
    num_points: usize,
) -> Result<Option<PathBuf>> {
    if config.staged_file_explicit || available_staged_export(config)?.is_some() {
        return resolve_staged_export(config, num_points).map(Some);
    }
    if cfg!(feature = "parlay") {
        return Ok(None);
    }
    Err(Error::unsupported("ParlayANN construction is unavailable; enable the parlay feature or provide a staged export")
        .at("build_graph").with_detail("backend", "parlayann"))
}

/// Resolve and validate the STAG header on a cache miss.
/// Cache hits skip this IO: the original export may have been removed.
pub(crate) fn resolve_staged_export(
    config: &ResolvedRunConfig,
    num_points: usize,
) -> Result<PathBuf> {
    let path = available_staged_export(config)?
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

pub(super) fn import_staged_graph<T: BaseElement>(
    config: &ResolvedRunConfig,
    dataset: PaddedDataset<T>,
    staged_path: &Path,
) -> Result<(Orion<T>, SourceFileIdentity)> {
    log::info!(
        "Streaming ParlayANN .staged export from {}",
        staged_path.display()
    );
    let num_points = dataset.len();
    // Preserve the existing sampled-medoid choice instead of the STAG hint.
    let entry = super::parlayann::sampled_medoid_rows(num_points, |id| dataset.row(id).unwrap());
    let file = std::fs::File::open(staged_path).map_err(|e| Error::io("open", staged_path, e))?;
    let bytes = file.metadata()?.len();
    let mut source = crate::integrity::HashedFile::new(file);
    let reader = std::io::BufReader::with_capacity(
        orion_ann_platform::buffered_io::FILE_WRITE_BYTES,
        &mut source,
    );
    let storage = crate::placement::graph_options(
        config,
        &dataset,
        crate::preflight::graph_bytes(config, num_points)?,
    )?;
    let index = Orion::from_staged_graph_with_storage(
        dataset,
        reader,
        bytes,
        config.graph_degree,
        entry,
        &storage,
    )
    .map_err(|e| {
        let message = format!("ParlayANN import failed: {}", e.message());
        e.with_message(message)
    })?;
    Ok((
        index,
        SourceFileIdentity::from_stream(staged_path, bytes, source.finish().1)?,
    ))
}
