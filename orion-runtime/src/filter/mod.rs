/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Filter request semantics and bounded resolution, independent of graph stages.
mod resolve;
use crate::{LabelKey, SearchResult};
pub(crate) use resolve::resolve;
use serde::{Deserialize, Serialize};

/// Exact labels combined as ALL(all) AND ANY(any) AND NOT ANY(none).
/// An empty `any` adds no restriction; an entirely empty filter permits all rows.
/// Unknown labels match nothing. No label predicate is evaluated in the ANN loop.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabelFilter {
    /// Labels every returned row must contain; an empty list imposes no conjunction.
    #[serde(default)]
    pub all: Vec<LabelKey>,

    /// Labels of which a row must contain at least one; an empty list imposes no disjunction.
    #[serde(default)]
    pub any: Vec<LabelKey>,

    /// Labels that disqualify a row even when it satisfies the positive terms.
    #[serde(default)]
    pub none: Vec<LabelKey>,
}
/// Controls for the initial exact filtered-search path. No adaptive calibration
/// is needed. This is an exhaustive baseline, not a filtered ANN performance claim.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FilterSearchOptions {
    /// Maximum eligible neighbors returned per query.
    pub k: usize,

    /// Whether to include original-metric scores for returned neighbors.
    pub scores: bool,
    /// Reject before vector evaluation if more rows match. Default: 100,000.
    pub max_candidates: usize,
}
impl FilterSearchOptions {
    /// Creates filtered-search controls with optional scoring disabled.
    pub const fn new(k: usize) -> Self {
        Self {
            k,
            scores: false,
            max_candidates: 100_000,
        }
    }

    /// Enables original-metric scores without changing eligibility or result selection.
    pub const fn with_scores(mut self, enabled: bool) -> Self {
        self.scores = enabled;
        self
    }
}
/// Results and the one label revision held for every query in the batch.
#[derive(Clone, Debug, PartialEq)]
pub struct FilteredSearchResult {
    /// Neighbors computed against one immutable allowlist generation.
    pub result: SearchResult,

    /// Published label revision used for the entire filtered request.
    pub label_revision: u64,
}
