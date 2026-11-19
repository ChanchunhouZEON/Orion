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

/// Persisted index recipe and references to immutable payloads; it contains no graph or vector buffers.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::index) struct Snapshot {
    /// Top-level snapshot schema version checked before decoding payload references.
    pub(in crate::index) format_version: u32,

    /// Location of the immutable content-addressed payload store.
    pub(in crate::index) object_store: PathBuf,

    /// Whether the portable bundle includes its original base object.
    #[serde(default)]
    pub(super) bundled_base: bool,

    /// Configuration needed to restore this index and its search representations.
    pub(super) options: IndexOptions,

    /// Prepared adaptive parameters keyed by their top-k target.
    pub(super) calibrations: BTreeMap<usize, Calibration>,

    /// Working-cache identity associated with this snapshot recipe.
    pub(super) namespace: String,

    /// Configuration signature checked when restoring search state.
    pub(super) execution_signature: String,

    /// Expected byte length of the referenced graph payload.
    pub(super) graph_bytes: u64,

    /// Expected byte length of the referenced graph metadata.
    pub(super) metadata_bytes: u64,

    /// Immutable object identities required to restore this generation.
    pub(in crate::index) payloads: Payloads,
}

/// Content identities needed to restore one coherent snapshot generation.
#[derive(Clone, Serialize, Deserialize)]
pub(in crate::index) struct Payloads {
    /// Optional supplier identity preserved with the graph payloads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supplier: Option<SupplierProvenance>,

    /// SHA-256 identity of the serialized graph payload.
    pub graph_sha256: String,

    /// SHA-256 identity of the graph metadata payload.
    pub metadata_sha256: String,

    /// SHA-256 identity of the recorded build/supplier provenance.
    pub provenance_sha256: String,

    /// Expected byte length of the provenance object.
    pub provenance_bytes: u64,

    /// Content identity binding the graph or snapshot to its original base.
    pub base_sha256: String,

    /// Content identities and sizes of all persisted sidecar representations.
    pub sidecars: SnapshotSidecars,

    /// Optional immutable mapping from internal IDs to application u64 IDs.
    pub external_ids: Option<ExternalIdObject>,

    /// Optional immutable label revision restored with this snapshot.
    #[serde(default)]
    pub labels: Option<crate::labels::LabelObject>,
}

/// Reference to the optional external-ID mapping stored alongside an index snapshot.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::index) struct ExternalIdObject {
    /// SHA-256 content identity used to locate and verify this object.
    pub sha256: String,

    /// Number of base vectors identified by this record.
    pub points: usize,
}

pub(in crate::index) const FORMAT_VERSION: u32 = 2;

/// Manifest records checked before engine creation. Graph and sidecar bytes
/// are verified by their decoders when the resident index is constructed.
pub(in crate::index) struct CheckedManifest {
    /// Verified graph metadata bytes retained for subsequent decoding.
    pub metadata: Vec<u8>,

    /// Resolved object-store location used for payload verification and loading.
    pub store: PathBuf,

    /// Immutable object identities required to restore this generation.
    pub payloads: Payloads,

    /// Prepared adaptive parameters keyed by their top-k target.
    pub calibrations: BTreeMap<usize, Calibration>,
}

impl Snapshot {
    /// Validates base identity, execution semantics, payload lengths and saved calibrations before restoration.
    ///
    /// Graph and sidecar contents are authenticated by their subsequent decoding reads.
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

/// Rejects fixed snapshot objects whose encoded sizes differ from the manifest.
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

/// Hashes the search-relevant configuration independently of deployment resource overrides.
pub(super) fn execution_signature(config: &ResolvedRunConfig) -> Result<String> {
    let bytes = serde_json::to_vec(&(
        config.cache_namespace.as_str(),
        config.metric.identity_key(),
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

    // Fixture includes metric-specific cosine/IP arms. Fixed paths/content IDs make this
    // independent of the host and test allocation order.
    #[test]
    fn resolved_config_and_snapshot_signatures_remain_stable() {
        let mut records = Vec::new();
        for (dimension, metric, storage, format) in [
            (128, SearchMetric::SquaredL2, None, VectorFormat::Fbin),
            (960, SearchMetric::SquaredL2, None, VectorFormat::Fbin),
            (
                128,
                SearchMetric::SquaredL2,
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

/// Supplier name and revision retained to describe an externally partitioned graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::index) struct SupplierProvenance {
    /// Supplier-defined algorithm name recorded as provenance.
    pub name: String,

    /// Optional supplier algorithm revision recorded with the graph.
    pub revision: Option<String>,
}

// Bind the exact topology and entry metadata, base, and search space. Supplier
// names are informational: identical payloads deduplicate across suppliers.
/// Derives a supplied graph namespace from its content identities and interpretation settings.
pub(in crate::index) fn supplier_namespace(
    config: &ResolvedRunConfig,
    files: &Payloads,
) -> Result<String> {
    Ok(crate::integrity::sha256(&serde_json::to_vec(&(
        "orion/supplied-graph",
        &files.base_sha256,
        &files.graph_sha256,
        &files.metadata_sha256,
        config.metric.identity_key(),
        config.storage(),
        config.dimension,
    ))?))
}
