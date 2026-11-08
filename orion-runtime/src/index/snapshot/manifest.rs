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

#[cfg(test)]
mod config_contract_tests {
    use super::*;
    use crate::data::VectorFormat;
    use crate::{SearchMetric, VectorStorageKind};

    // Captured before reorganizing config. Fixed paths/content IDs make this
    // independent of the host and test allocation order.
    #[test]
    fn resolved_config_and_snapshot_signatures_remain_stable() {
        let mut records = Vec::new();
        for (dimension, metric, storage, format) in [
            (128, SearchMetric::L2, None, VectorFormat::Fbin),
            (960, SearchMetric::L2, None, VectorFormat::Fbin),
            (
                128,
                SearchMetric::L2,
                Some(VectorStorageKind::U8),
                VectorFormat::U8bin,
            ),
            (100, SearchMetric::InnerProduct, None, VectorFormat::Fbin),
            (1536, SearchMetric::Cosine, None, VectorFormat::Fbin),
        ] {
            let options = IndexOptions {
                metric,
                storage,
                threads: 8,
                ..Default::default()
            };
            let config = options
                .resolve(
                    "fixture/base".into(),
                    format,
                    dimension,
                    1000,
                    Some("fixture-content".into()),
                )
                .unwrap();
            records.push(serde_json::json!({
                "options": options,
                "resolved": config,
                "execution_signature": execution_signature(&config).unwrap(),
                "resolved_wire_sha256": crate::integrity::sha256(&serde_json::to_vec(&config).unwrap()),
                "options_wire_sha256": crate::integrity::sha256(&serde_json::to_vec(&options).unwrap()),
            }));
        }
        let actual = serde_json::Value::Array(records);
        let expected: serde_json::Value =
            serde_json::from_str(include_str!("../../config/tests/resolution.json")).unwrap();
        assert_eq!(actual, expected);
    }
}
