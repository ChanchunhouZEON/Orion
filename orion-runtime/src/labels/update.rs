/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Batch input and validation; publication belongs to the store.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// An exact, case-sensitive (namespace, value) pair. No implicit normalization.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabelKey {
    pub namespace: String,
    pub value: String,
}
impl LabelKey {
    /// Both strings must be nonempty; their combined UTF-8 length is <= 1024 bytes.
    pub fn new(namespace: impl Into<String>, value: impl Into<String>) -> Result<Self> {
        let key = Self {
            namespace: namespace.into(),
            value: value.into(),
        };
        key.validate()?;
        Ok(key)
    }
    pub(crate) fn validate(&self) -> Result<()> {
        if self.namespace.is_empty()
            || self.value.is_empty()
            || self.namespace.len().saturating_add(self.value.len()) > super::MAX_KEY_BYTES
        {
            return Err(Error::invalid_argument(
                "label requires nonempty namespace/value, at most 1024 UTF-8 bytes combined",
            ));
        }
        Ok(())
    }
}

/// Add/remove internal row IDs in one label. A batch may mention a label once.
/// Duplicate IDs within either list are idempotent; overlapping add/remove IDs
/// are rejected. Empty labels are removed. No vectors or graph edges are changed.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabelUpdate {
    pub label: LabelKey,
    #[serde(default)]
    pub add: Vec<u32>,
    #[serde(default)]
    pub remove: Vec<u32>,
}

/// Add/remove business IDs in one label, resolved by the owning Index.
///
/// Use with [`crate::Index::update_labels_external`]. IDs are u64 business IDs,
/// never internal row numbers. The whole batch is converted before publication;
/// an unknown ID (including a remove ID) rejects every change in that batch.
/// Duplicate and overlapping IDs follow the same rules as [`LabelUpdate`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalLabelUpdate {
    pub label: LabelKey,
    #[serde(default)]
    pub add: Vec<u64>,
    #[serde(default)]
    pub remove: Vec<u64>,
}

/// Apply identical input budgets before allocating internal or converted batches.
pub(crate) fn check_batch_limits(
    count: usize,
    mut sizes: impl Iterator<Item = (usize, usize)>,
) -> Result<()> {
    if count > super::MAX_LABELS
        || sizes
            .try_fold(0usize, |n, (add, remove)| {
                n.checked_add(add)?.checked_add(remove)
            })
            .is_none_or(|n| n > super::MAX_MEMBERSHIPS)
    {
        return Err(Error::invalid_argument("label update exceeds batch limits"));
    }
    Ok(())
}

pub(super) fn checked_ids(ids: &[u32], points: usize) -> Result<Vec<u32>> {
    if ids.iter().any(|&id| id as usize >= points) {
        return Err(Error::invalid_argument(
            "label row ID is outside this index",
        ));
    }
    let mut out = ids.to_vec();
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// Sorted union without allocating one tree node per membership.
pub(crate) fn union(a: &[u32], b: &[u32]) -> Vec<u32> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::with_capacity(a.len().saturating_add(b.len()));
    while i < a.len() || j < b.len() {
        if j == b.len() || (i < a.len() && a[i] < b[j]) {
            out.push(a[i]);
            i += 1;
        } else if i == a.len() || b[j] < a[i] {
            out.push(b[j]);
            j += 1;
        } else {
            out.push(a[i]);
            i += 1;
            j += 1;
        }
    }
    out
}
pub(crate) fn difference(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter()
        .copied()
        .filter(|id| b.binary_search(id).is_err())
        .collect()
}
