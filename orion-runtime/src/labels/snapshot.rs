/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! A query owns one immutable label generation for its entire batch.
use super::LabelKey;
use std::{collections::BTreeMap, sync::Arc};
#[derive(Debug)]
pub(crate) struct LabelSnapshot {
    pub identity: String,
    pub revision: u64,
    pub points: usize,
    pub labels: BTreeMap<LabelKey, Arc<Vec<u32>>>,
}
