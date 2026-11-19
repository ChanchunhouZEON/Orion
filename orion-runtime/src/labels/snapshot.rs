/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! A query owns one immutable label generation for its entire batch.
use super::LabelKey;
use std::{collections::BTreeMap, sync::Arc};

/// Immutable label membership generation retained by searches and snapshot saves.
#[derive(Debug)]
pub(crate) struct LabelSnapshot {
    /// Digest binding membership to this base and external-ID mapping.
    pub identity: String,

    /// Monotonic publication revision within the bound label identity space.
    pub revision: u64,

    /// Internal-ID universe against which all memberships are validated.
    pub points: usize,

    /// Sorted memberships shared between revisions when a label is unchanged.
    pub labels: BTreeMap<LabelKey, Arc<Vec<u32>>>,
}
