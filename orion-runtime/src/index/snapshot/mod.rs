/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resolve a snapshot directory or trusted reference and open its generation.
mod bundle;
pub(super) mod manifest;
mod save;
use crate::{BaseInput, Error, Index, Result};
pub(super) use manifest::{CheckedManifest, ExternalIdObject, Payloads, Snapshot};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
/// A snapshot's location and the manifest identity recorded when it was saved.
/// Serialize this reference in trusted application state to reopen after a restart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotRef {
    path: PathBuf,
    manifest_sha256: String,
}

impl SnapshotRef {
    /// Retain the manifest identity after moving a snapshot or bundle directory.
    pub fn relocated(&self, path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            manifest_sha256: self.manifest_sha256.clone(),
        }
    }

    /// Borrows the snapshot directory recorded by this saved reference.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the expected manifest digest used to detect replacement of this snapshot record.
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
}

/// Open the snapshot currently in a directory, or require the identity issued by save.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SnapshotSource {
    Directory(PathBuf),
    Reference(SnapshotRef),
}

impl<P: Into<PathBuf>> From<P> for SnapshotSource {
    /// Records a snapshot directory for later verification and loading.
    fn from(path: P) -> Self {
        Self::Directory(path.into())
    }
}

impl From<SnapshotRef> for SnapshotSource {
    /// Takes the snapshot reference without opening or verifying its payloads.
    fn from(reference: SnapshotRef) -> Self {
        Self::Reference(reference)
    }
}

impl From<&SnapshotRef> for SnapshotSource {
    /// Copies the snapshot reference identity without opening files.
    fn from(reference: &SnapshotRef) -> Self {
        Self::Reference(reference.clone())
    }
}

/// Display and resource overrides for reopening a snapshot. Defaults use the saved settings.
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpenOptions {
    /// Override the saved display name without changing payload or calibration identity.
    pub dataset_name: Option<String>,
    /// Explicit relocation of the immutable object store. Hash verification is unchanged.
    pub object_store: Option<PathBuf>,

    /// Optional worker-count override for the restored index.
    pub threads: Option<usize>,

    /// Optional managed resident-payload budget for restoration, expressed in GiB.
    pub memory_budget_gib: Option<f64>,
    /// Override the saved deployment policy; resources are always probed afresh.
    pub auto_pin: Option<bool>,

    /// Optional per-object placement policy for the restored payloads.
    pub storage_options: Option<crate::StorageOptions>,
}

impl OpenOptions {
    /// Open through the same integrity checks as [`Index::open`].
    pub fn open(self, source: impl Into<SnapshotSource>, input: BaseInput) -> Result<Index> {
        let (path, snapshot) = read_manifest(source.into())?;
        self.load_snapshot(&path, snapshot, input)
    }

    /// Resolves object-store relocation and deployment overrides before loading a verified snapshot.
    ///
    /// Saved graph and calibration semantics remain intact; resource choices may differ
    /// from those used by the machine that created the snapshot.
    fn load_snapshot(self, path: &Path, mut snapshot: Snapshot, input: BaseInput) -> Result<Index> {
        snapshot.object_store = match self.object_store {
            Some(store) => store,
            None if snapshot.object_store.is_relative() => path.join(&snapshot.object_store),
            None => snapshot.object_store,
        }
        .canonicalize()
        .map_err(Error::from)?;
        let mut options = snapshot.options.clone();
        options.cache_dir = path.canonicalize().map_err(Error::from)?;
        if self.dataset_name.is_some() {
            options.dataset_name = self.dataset_name;
        }
        if let Some(n) = self.threads {
            options.threads = n;
        }
        if self.memory_budget_gib.is_some() {
            options.memory_budget_gib = self.memory_budget_gib;
        }
        if let Some(auto_pin) = self.auto_pin {
            options.auto_pin = auto_pin;
        }
        if let Some(storage) = self.storage_options {
            options.storage_options = storage;
        }
        Index::load(input, options, Some(snapshot))
    }
}

/// Parse once so selecting a bundled base and loading payloads use the same
/// checked manifest, even if another process replaces the directory meanwhile.
fn read_manifest(source: SnapshotSource) -> Result<(PathBuf, Snapshot)> {
    let path = match &source {
        SnapshotSource::Directory(path) => path.clone(),
        SnapshotSource::Reference(reference) => reference.path().to_owned(),
    };
    let bytes = crate::storage::read_bounded(&path.join("index.json"), 16 * 1024 * 1024)?;
    if let SnapshotSource::Reference(reference) = source {
        crate::integrity::check(
            "manifest",
            &crate::integrity::sha256(&bytes),
            reference.manifest_sha256(),
        )?;
    }
    let envelope: serde_json::Value = serde_json::from_slice(&bytes)?;
    let version = envelope["format_version"].as_u64();
    // V1 predates labels. A labelled manifest must use V2 so old readers reject
    // it instead of silently restoring an index without its filtering state.
    let legacy = version == Some(1) && envelope["payloads"]["labels"].is_null();
    if !legacy && version != Some(manifest::FORMAT_VERSION as u64) {
        return Err(Error::cache(
            "unsupported snapshot format version; export with a compatible reader",
        )
        .with_detail("supported_version", manifest::FORMAT_VERSION));
    }
    Ok((path, serde_json::from_value(envelope)?))
}
