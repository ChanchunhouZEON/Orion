/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Verified immutable label objects. Saving never modifies the live revision.
use super::{LabelKey, LabelSnapshot, MAX_LABELS, MAX_MEMBERSHIPS};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path, sync::Arc};
pub(crate) const CODEC: &str = "labels.json";
const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LabelObject {
    pub sha256: String,
    pub revision: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    format_version: u32,
    identity: String,
    points: usize,
    revision: u64,
    labels: Vec<(LabelKey, Vec<u32>)>,
}

pub(crate) fn identity(base: &str, points: usize, external_ids: Option<&str>) -> String {
    // Separates equal-length datasets and different business-ID assignments.
    // Graph/cascade/calibration changes do not alter the row identity space.
    crate::integrity::sha256(
        &serde_json::to_vec(&(base, points, external_ids)).expect("identity tuple is serializable"),
    )
}
impl LabelSnapshot {
    pub(crate) fn save(&self, store: &Path) -> Result<LabelObject> {
        // Arc payloads are borrowed during serialization; no membership clone.
        #[derive(Serialize)]
        struct Borrowed<'a> {
            format_version: u32,
            identity: &'a str,
            points: usize,
            revision: u64,
            labels: Vec<(&'a LabelKey, &'a [u32])>,
        }
        let wire = Borrowed {
            format_version: 1,
            identity: &self.identity,
            points: self.points,
            revision: self.revision,
            labels: self
                .labels
                .iter()
                .map(|(key, ids)| (key, ids.as_slice()))
                .collect(),
        };
        let (_, sha256) = crate::storage::put(store, CODEC, |writer| {
            serde_json::to_writer(writer, &wire).map_err(Error::from)
        })?;
        Ok(LabelObject {
            sha256,
            revision: self.revision,
        })
    }
    pub(crate) fn load(
        store: &Path,
        object: &LabelObject,
        expected_identity: &str,
        points: usize,
    ) -> Result<Self> {
        let bytes =
            crate::storage::read_record(store, &object.sha256, CODEC, MAX_FILE_BYTES, "labels")?;
        let record: Record = serde_json::from_slice(&bytes)
            .map_err(|e| Error::cache(format!("invalid label object: {e}")))?;
        if record.format_version != 1
            || record.identity != expected_identity
            || record.points != points
            || record.revision != object.revision
        {
            return Err(Error::cache(
                "label object version, identity, shape or revision mismatch",
            ));
        }
        if record.labels.len() > MAX_LABELS
            || record
                .labels
                .iter()
                .map(|(_, ids)| ids.len())
                .sum::<usize>()
                > MAX_MEMBERSHIPS
        {
            return Err(Error::cache("label object exceeds metadata limits"));
        }
        let mut labels = BTreeMap::new();
        for (key, ids) in record.labels {
            key.validate()
                .map_err(|_| Error::cache("invalid label key"))?;
            if ids.is_empty()
                || ids.iter().any(|&id| id as usize >= points)
                || ids.windows(2).any(|w| w[0] >= w[1])
                || labels.insert(key, Arc::new(ids)).is_some()
            {
                return Err(Error::cache(
                    "label object has duplicate keys or invalid membership IDs",
                ));
            }
        }
        if record.revision == 0 && !labels.is_empty() {
            return Err(Error::cache("initial label revision must be empty"));
        }
        Ok(Self {
            identity: record.identity,
            points,
            revision: record.revision,
            labels,
        })
    }
}
