/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Snapshot wire format and validation of manifest fields and small records.
//! Snapshot is the serialized input, Payloads names its immutable objects, and
//! CheckedManifest is the validated result passed to resident preparation.
use crate::index::{query::validate_search, Calibration};
use crate::{
    config::ResolvedRunConfig,
    integrity::check,
    sidecars::SnapshotSidecars,
    storage::{object, read_record},
    Error, IndexOptions, Result,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::index) struct Snapshot {
    #[serde(default)]
    pub(in crate::index) object_store: PathBuf,
    pub(super) options: IndexOptions,
    pub(super) calibrations: BTreeMap<usize, Calibration>,
    pub(super) namespace: String,
    pub(super) execution_signature: String,
    pub(super) graph_bytes: u64,
    pub(super) metadata_bytes: u64,
    pub(in crate::index) payloads: Payloads,
}

#[derive(Clone, Serialize, Deserialize)]
pub(in crate::index) struct Payloads {
    pub graph_sha256: String,
    pub metadata_sha256: String,
    pub provenance_sha256: String,
    pub provenance_bytes: u64,
    pub base_sha256: String,
    pub sidecars: SnapshotSidecars,
}

/// Manifest records checked before engine creation. Graph and sidecar bytes
/// are verified by their decoders when the resident index is constructed.
pub(in crate::index) struct CheckedManifest {
    pub metadata: Vec<u8>,
    pub store: PathBuf,
    pub payloads: Payloads,
    pub calibrations: BTreeMap<usize, Calibration>,
}

impl Snapshot {
    pub(in crate::index) fn verify(
        self,
        config: &ResolvedRunConfig,
        points: usize,
        base_sha256: &str,
    ) -> Result<CheckedManifest> {
        check("base identity", base_sha256, &self.payloads.base_sha256)?;
        if self.namespace != config.cache_namespace
            || self.execution_signature != execution_signature(config)?
        {
            return Err(
                Error::cache("cache base identity or graph configuration mismatch").at("open"),
            );
        }
        let store = self.object_store;
        let files = self.payloads;
        let metadata = read_record(&store, &files.metadata_sha256, "bin", 4096, "metadata")?;
        read_record(
            &store,
            &files.provenance_sha256,
            "cache.json",
            1024 * 1024,
            "provenance",
        )?;
        check_lengths(&store, &files, self.graph_bytes, self.metadata_bytes)?;
        for (&k, c) in &self.calibrations {
            validate_search(points, k, c.search_list_size)?;
            if k != c.k
                || c.samples == 0
                || !c.threshold.is_finite()
                || !(0.0..=1.0).contains(&c.threshold)
                || c.early_exit_limit == 0
            {
                return Err(Error::cache("invalid saved calibration").at("open"));
            }
        }
        Ok(CheckedManifest {
            metadata,
            store,
            payloads: files,
            calibrations: self.calibrations,
        })
    }
}

pub(in crate::index) fn check_lengths(
    store: &Path,
    files: &Payloads,
    graph_bytes: u64,
    metadata_bytes: u64,
) -> Result<()> {
    let fixed = [
        ("bin", metadata_bytes, &files.metadata_sha256),
        ("pgraph", graph_bytes, &files.graph_sha256),
        (
            "cache.json",
            files.provenance_bytes,
            &files.provenance_sha256,
        ),
    ];
    for (ext, len, digest) in fixed {
        if fs::metadata(object(store, digest, ext)?)?.len() != len {
            return Err(Error::cache("snapshot payload length mismatch").with_detail("role", ext));
        }
    }
    Ok(())
}

pub(super) fn execution_signature(config: &ResolvedRunConfig) -> Result<String> {
    let bytes = serde_json::to_vec(&(
        config.cache_namespace.as_str(),
        config.metric,
        config.index_plan(),
        config.window_size,
    ))
    .map_err(Error::from)?;
    Ok(crate::integrity::sha256(&bytes))
}
